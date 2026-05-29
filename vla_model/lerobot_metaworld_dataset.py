from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .data import SequenceConfig, action_slice_from_frame_t, pad_frame_indices, valid_tick_indices
from .scheduler import ControlScheduler
from .schema import VLAConfig
from .task_splits import DATASET_TASK_TO_SLUG


def _norm_actions(actions: Tensor, stats: dict, key: str = "action") -> Tensor:
    if key not in stats:
        return actions
    info = stats[key]
    if "min" in info and "max" in info:
        mn = torch.tensor(info["min"], dtype=actions.dtype)
        mx = torch.tensor(info["max"], dtype=actions.dtype)
        scale = (mx - mn).clamp_min(1e-6)
        return 2.0 * (actions - mn) / scale - 1.0
    if "mean" in info and "std" in info:
        mean = torch.tensor(info["mean"], dtype=actions.dtype)
        std = torch.tensor(info["std"], dtype=actions.dtype).clamp_min(1e-6)
        return (actions - mean) / std
    return actions


def _norm_states(states: Tensor, stats: dict, key: str = "observation.state") -> Tensor:
    if key not in stats:
        return states
    info = stats[key]
    if "mean" in info and "std" in info:
        mean = torch.tensor(info["mean"], dtype=states.dtype)
        std = torch.tensor(info["std"], dtype=states.dtype).clamp_min(1e-6)
        return (states - mean) / std
    return states


@dataclass
class LeRobotDatasetConfig:
    root_dir: str
    split: str = "train"
    val_ratio: float = 0.1
    seed: int = 42
    hard_tasks: Optional[List[str]] = None
    max_episodes: Optional[int] = None
    frame_key: str = "observation.images.image"
    states_key: str = "observation.state"
    action_key: str = "action"
    use_stats_norm: bool = True
    episode_cache_size: int = 8
    decode_video: bool = True


@dataclass
class LeRobotEpisodeRef:
    episode_id: int
    task_name: str
    parquet_path: Path
    chunk_name: str
    num_frames: int
    num_actions: int


class LeRobotMetaWorldDataset(Dataset):
    def __init__(self, cfg: LeRobotDatasetConfig, model_cfg: VLAConfig, seq_cfg: SequenceConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.seq_cfg = seq_cfg
        self.scheduler = ControlScheduler(model_cfg.scheduler)
        self.root = Path(cfg.root_dir)
        self.meta_dir = self.root / "meta"
        self.data_dir = self.root / "data"
        self.video_dir = self.root / "videos"
        self.stats = self._load_stats()
        self.task_map = self._load_tasks()
        self.episodes = self._scan_episodes()
        self.index = self._build_index()
        self._episode_cache: Dict[int, Dict[str, Tensor]] = {}
        self._episode_order: List[int] = []

    def _load_stats(self) -> dict:
        stats_path = self.meta_dir / "stats.json"
        if not stats_path.exists():
            return {}
        with stats_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_tasks(self) -> Dict[int, str]:
        path = self.meta_dir / "tasks.jsonl"
        if not path.exists():
            return {}
        mapping: Dict[int, str] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                tid = int(obj.get("task_index", obj.get("task_id", -1)))
                mapping[tid] = str(obj.get("task", obj.get("task_name", "unknown")))
        return mapping

    def _episode_iter(self) -> List[LeRobotEpisodeRef]:
        meta_path = self.meta_dir / "episodes.jsonl"
        refs: List[LeRobotEpisodeRef] = []
        if meta_path.exists():
            # Build reverse map: task_name → task_index
            task_name_to_idx: dict[str, int] = {}
            for tid, tname in self.task_map.items():
                task_name_to_idx[tname] = tid

            with meta_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    episode_id = int(obj.get("episode_index", obj.get("episode_id", obj.get("id", -1))))
                    chunk = str(obj.get("chunk", "chunk-000"))
                    parquet = self.data_dir / chunk / f"episode_{episode_id:06d}.parquet"
                    if not parquet.exists():
                        continue
                    tasks_arr = obj.get("tasks", None)
                    if tasks_arr is not None and isinstance(tasks_arr, list) and len(tasks_arr) > 0:
                        task_name = str(tasks_arr[0])
                    else:
                        task_idx = int(obj.get("task_index", -1))
                        task_name = self.task_map.get(task_idx, "unknown")
                    num_frames = int(obj.get("length", obj.get("num_steps", -1)))
                    refs.append(
                        LeRobotEpisodeRef(
                            episode_id=episode_id,
                            task_name=task_name,
                            parquet_path=parquet,
                            chunk_name=chunk,
                            num_frames=num_frames,
                            num_actions=-1,
                        )
                    )
        if refs:
            return refs
        for p in sorted(self.data_dir.glob("chunk-*/*.parquet")):
            try:
                episode_id = int(p.stem.split("_")[-1])
            except ValueError:
                continue
            refs.append(
                LeRobotEpisodeRef(
                    episode_id=episode_id,
                    task_name="unknown",
                    parquet_path=p,
                    chunk_name=p.parent.name,
                    num_frames=-1,
                    num_actions=-1,
                )
            )
        return refs

    def _scan_episodes(self) -> List[LeRobotEpisodeRef]:
        refs = self._episode_iter()
        if self.cfg.hard_tasks:
            hard_set = set(self.cfg.hard_tasks)
            matched = []
            for r in refs:
                slug = DATASET_TASK_TO_SLUG.get(r.task_name, r.task_name)
                if slug in hard_set:
                    matched.append(r)
            refs = matched
        if self.cfg.max_episodes is not None:
            refs = refs[: self.cfg.max_episodes]
        if not refs:
            raise RuntimeError("No episodes found for LeRobot dataset.")
        filled: List[LeRobotEpisodeRef] = []
        for ref in refs:
            df = pd.read_parquet(ref.parquet_path, columns=[self.cfg.action_key])
            num_actions = len(df)
            num_frames = ref.num_frames if ref.num_frames > 0 else max(1, num_actions // self.seq_cfg.actions_per_frame)
            filled.append(
                LeRobotEpisodeRef(
                    episode_id=ref.episode_id,
                    task_name=ref.task_name,
                    parquet_path=ref.parquet_path,
                    chunk_name=ref.chunk_name,
                    num_frames=num_frames,
                    num_actions=num_actions,
                )
            )
        rng = np.random.default_rng(self.cfg.seed)
        order = np.arange(len(filled))
        rng.shuffle(order)
        split = int((1.0 - self.cfg.val_ratio) * len(order))
        keep = set(order[:split].tolist()) if self.cfg.split == "train" else set(order[split:].tolist())
        out = [r for i, r in enumerate(filled) if i in keep]
        if not out:
            raise RuntimeError(f"No episodes in split={self.cfg.split}")
        return out

    def _build_index(self) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for epi_id, ep in enumerate(self.episodes):
            for frame_t in valid_tick_indices(ep.num_frames, ep.num_actions, self.seq_cfg):
                out.append((epi_id, frame_t))
        if not out:
            raise RuntimeError("No valid JEPA tick samples from LeRobot dataset.")
        return out

    def __len__(self) -> int:
        return len(self.index)

    def _video_path(self, ep: LeRobotEpisodeRef) -> Path:
        return self.video_dir / ep.chunk_name / self.cfg.frame_key / f"episode_{ep.episode_id:06d}.mp4"

    def _decode_video(self, path: Path) -> Tensor:
        import av

        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        container = av.open(str(path))
        stream = container.streams.video[0]
        frames = [torch.from_numpy(frame.to_ndarray(format="rgb24")) for frame in container.decode(stream)]
        container.close()
        if not frames:
            raise RuntimeError(f"No frames decoded from {path}")
        return torch.stack(frames).float() / 255.0

    def _load_episode_arrays(self, epi_idx: int) -> Dict[str, Tensor]:
        if epi_idx in self._episode_cache:
            return self._episode_cache[epi_idx]
        ep = self.episodes[epi_idx]
        df = pd.read_parquet(ep.parquet_path)
        states = torch.tensor(np.stack(df[self.cfg.states_key].to_list()), dtype=torch.float32)
        actions = torch.tensor(np.stack(df[self.cfg.action_key].to_list()), dtype=torch.float32)
        if self.cfg.use_stats_norm:
            states = _norm_states(states, self.stats, key=self.cfg.states_key)
            actions = _norm_actions(actions, self.stats, key=self.cfg.action_key)
        arr: Dict[str, Tensor] = {"states": states, "actions": actions}
        if self.cfg.decode_video:
            arr["images"] = self._decode_video(self._video_path(ep))
        self._episode_cache[epi_idx] = arr
        self._episode_order.append(epi_idx)
        while len(self._episode_order) > self.cfg.episode_cache_size:
            old = self._episode_order.pop(0)
            self._episode_cache.pop(old, None)
        return arr

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        epi_idx, frame_t = self.index[idx]
        arrays = self._load_episode_arrays(epi_idx)
        if "images" not in arrays:
            raise RuntimeError("decode_video=False is not supported for the new JEPA pipeline.")
        images = arrays["images"]
        states = arrays["states"]
        actions = arrays["actions"]

        context_frames = images[frame_t - self.seq_cfg.jepa_window_frames + 1 : frame_t + 1]
        sparse_idx = pad_frame_indices(frame_t, self.seq_cfg.robot_hist_frame_offsets)
        sparse_hist_images = images[sparse_idx]
        sparse_hist_states = states[sparse_idx]

        sparse_hist_actions = []
        for hist_frame_idx in sparse_idx:
            action_idx = min(hist_frame_idx * self.seq_cfg.actions_per_frame, actions.shape[0] - 1)
            sparse_hist_actions.append(actions[action_idx])
        sparse_hist_actions_t = torch.stack(sparse_hist_actions, dim=0)

        target_slice = action_slice_from_frame_t(frame_t, self.seq_cfg.action_horizon, self.seq_cfg.actions_per_frame)
        target_actions = actions[target_slice]
        if target_actions.shape[0] != self.seq_cfg.action_horizon:
            raise RuntimeError(
                f"Episode {self.episodes[epi_idx].parquet_path} produced invalid target horizon {target_actions.shape[0]}"
            )

        tick = self.scheduler.tick_info(frame_t)
        return {
            "context_frames": context_frames,
            "sparse_hist_images": sparse_hist_images,
            "sparse_hist_states": sparse_hist_states,
            "sparse_hist_actions": sparse_hist_actions_t,
            "target_actions": target_actions,
            "frame_idx": torch.tensor(frame_t, dtype=torch.long),
            "action_step_idx": torch.tensor(tick.action_step_idx, dtype=torch.long),
            "jepa_tick_idx": torch.tensor(tick.jepa_tick_idx, dtype=torch.long),
        }
