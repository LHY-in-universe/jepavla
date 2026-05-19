from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .schema import VLAConfig
from .data import SequenceConfig
from .metaworld_dataset import RandomProjectionFeatureizer


def _norm_actions(actions: Tensor, stats: dict, key: str = "action") -> Tensor:
    if key not in stats:
        return actions
    info = stats[key]
    if "min" in info and "max" in info:
        mn = torch.tensor(info["min"], dtype=actions.dtype, device=actions.device)
        mx = torch.tensor(info["max"], dtype=actions.dtype, device=actions.device)
        scale = (mx - mn).clamp_min(1e-6)
        # map to [-1, 1]
        return 2.0 * (actions - mn) / scale - 1.0
    if "mean" in info and "std" in info:
        mean = torch.tensor(info["mean"], dtype=actions.dtype, device=actions.device)
        std = torch.tensor(info["std"], dtype=actions.dtype, device=actions.device).clamp_min(1e-6)
        return (actions - mean) / std
    return actions


def _norm_states(states: Tensor, stats: dict, key: str = "observation.state") -> Tensor:
    if key not in stats:
        return states
    info = stats[key]
    if "mean" in info and "std" in info:
        mean = torch.tensor(info["mean"], dtype=states.dtype, device=states.device)
        std = torch.tensor(info["std"], dtype=states.dtype, device=states.device).clamp_min(1e-6)
        return (states - mean) / std
    return states


class _VideoDecoderLRU:
    """Reusable video decoder with LRU cache, exactly matching Evo-1 approach."""

    def __init__(self, backend: str = "av", max_size: int = 16) -> None:
        self.backend = backend
        self.max_size = max(1, int(max_size))
        self._cache: "Dict[Path, Any]" = {}
        self._order: "List[Path]" = []

    def _evict_if_needed(self):
        while len(self._cache) > self.max_size:
            path = self._order.pop(0)
            handle = self._cache.pop(path, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass

    def _get_pyav_container(self, path: Path):
        import av
        if path in self._cache:
            self._order.remove(path)
            self._order.append(path)
            return self._cache[path]
        container = av.open(str(path))
        self._cache[path] = container
        self._order.append(path)
        self._evict_if_needed()
        return container

    def close_all(self):
        for _, handle in self._cache.items():
            try:
                handle.close()
            except Exception:
                pass
        self._cache.clear()
        self._order.clear()

    def decode_single_frame(self, path: Path, timestamp: float):
        """Decode a single frame at given timestamp. Returns PIL Image (matches Evo-1)."""
        from PIL import Image
        container = self._get_pyav_container(path)
        stream = container.streams.video[0]
        try:
            container.seek(0, stream=stream)
            last_frame = None
            for frame in container.decode(video=0):
                last_frame = frame
                if frame.time is not None and frame.time >= timestamp:
                    return Image.fromarray(frame.to_ndarray(format="rgb24"))
            if last_frame is None:
                raise RuntimeError(f"No frame decoded from {path}")
            return Image.fromarray(last_frame.to_ndarray(format="rgb24"))
        except Exception:
            if path in self._cache:
                try:
                    self._cache[path].close()
                except Exception:
                    pass
                del self._cache[path]
            raise


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
    episode_cache_size: int = 16
    decode_video: bool = True


@dataclass
class LeRobotEpisodeRef:
    episode_id: int
    task_name: str
    parquet_path: Path
    chunk_name: str
    num_steps: int


class LeRobotMetaWorldDataset(Dataset):
    """
    LeRobot v2.1 style dataset adapter for MetaWorld offline training.

    Layout:
    - data/chunk-*/episode_*.parquet
    - videos/chunk-*/<frame_key>/episode_*.mp4
    - meta/tasks.jsonl
    - meta/episodes.jsonl
    - meta/stats.json (optional)
    """

    def __init__(self, cfg: LeRobotDatasetConfig, model_cfg: VLAConfig, seq_cfg: SequenceConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.seq_cfg = seq_cfg
        self.featureizer = RandomProjectionFeatureizer(model_cfg, seed=cfg.seed, device="cuda")

        self.root = Path(cfg.root_dir)
        self.meta_dir = self.root / "meta"
        self.data_dir = self.root / "data"
        self.video_dir = self.root / "videos"

        self.task_map = self._load_tasks()
        self.stats = self._load_stats()
        self.episodes = self._scan_episodes()
        self.index = self._build_index()

        self._episode_cache: Dict[int, Dict[str, Tensor]] = {}
        self._episode_order: List[int] = []

        # Qwen3-VL pre-extracted features
        self._vl_dir = self.root / "vl_features_qwen3"
        self._vl_cache: Dict[int, Tensor] = {}
        self._vl_dim = 2048  # Qwen3-VL-2B vision dim

    def _load_stats(self) -> dict:
        stats_path = self.meta_dir / "stats.json"
        if not stats_path.exists():
            return {}
        with stats_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_tasks(self) -> Dict[int, str]:
        """Load task mapping: task_index -> slug (e.g., 'nut-assembly-v3').
        
        We map language instructions to MetaWorld MT50 slugs because:
        - hard_tasks filter uses slug format (e.g., 'nut-assembly-v3')
        - task_name must match these slugs for filtering to work
        """
        path = self.meta_dir / "tasks.jsonl"
        if not path.exists():
            return {}
        
        # Language to slug mapping for MetaWorld MT50
        LANGUAGE_TO_SLUG = {
            "Pick up a nut and place it onto a peg": "nut-assembly-v3",
            "Push a mug under a coffee machine": "coffee-button-v3",
            "Rotate a dial 180 degrees": "dial-turn-v3",
            "Pick a nut out of a peg": "nut-disassemble-v3",
            "Close a door with a revolving joint": "door-close-v3",
            "Lock the door by rotating the lock clockwise": "door-lock-v3",
            "Open a door with a revolving joint": "door-open-v3",
            "Unlock the door by rotating the lock counter-clockwise": "door-unlock-v3",
            "Insert the gripper into a hole": "hand-insert-v3",
            "Push and close a drawer": "drawer-close-v3",
            "Open a drawer": "drawer-open-v3",
            "Dunk the basketball into the basket": "basketball-v3",
            "Rotate the faucet counter-clockwise": "faucet-open-v3",
            "Rotate the faucet clockwise": "faucet-close-v3",
            "Hammer a screw on the wall": "hammer-v3",
            "Press a handle down sideways": "handle-press-side-v3",
            "Press a handle down": "handle-press-v3",
            "Pull a handle up sideways": "handle-pull-side-v3",
            "Pull a handle up": "handle-pull-v3",
            "Pull a lever down 90 degrees": "lever-pull-v3",
            "Pick a puck, bypass a wall and place the puck": "pick-place-wall-v3",
            "Pick up a puck from a hole": "pick-out-of-hole-v3",
            "Grasp the puck from one bin and place it into another bin": "bin-picking-v3",
            "Pick and place a puck to a goal": "pick-place-v3",
            "Slide a plate into a cabinet": "plate-slide-v3",
            "Slide a plate into a cabinet sideways": "plate-slide-side-v3",
            "Get a plate from the cabinet": "plate-slide-back-v3",
            "Get a plate from the cabinet sideways": "plate-slide-back-side-v3",
            "Insert a peg sideways": "peg-insertion-side-v3",
            "Unplug a peg sideways": "peg-unplug-side-v3",
            "Kick a soccer into the goal": "soccer-v3",
            "Grasp a stick and push a box using the stick": "stick-push-v3",
            "Grasp a stick and pull a box with the stick": "stick-pull-v3",
            "Grasp the cover and close the box with it": "box-close-v3",
            "Push the puck to a goal": "push-v3",
            "Bypass a wall and push a puck to a goal": "push-wall-v3",
            "Push the puck back to a goal": "push-back-v3",
            "Reach a goal position": "reach-v3",
            "Bypass a wall and reach a goal": "reach-wall-v3",
            "Pick and place a puck onto a shelf": "shelf-place-v3",
            "Sweep a puck into a hole": "sweep-into-goal-v3",
            "Sweep a puck off the table": "sweep-v3",
            "Push and open a window": "window-open-v3",
            "Push and close a window": "window-close-v3",
            "Press a button from the top": "button-press-topdown-v3",
            "Bypass a wall and press a button from the top": "button-press-topdown-wall-v3",
            "Press a button": "button-press-v3",
            "Bypass a wall and press a button": "button-press-wall-v3",
            "Push a button on the coffee machine": "coffee-pull-v3",
            "Pull a mug from a coffee machine": "coffee-push-v3",
        }
        
        task_map: Dict[int, str] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                tid = int(obj.get("task_index", obj.get("task_id", -1)))
                task_lang = str(obj.get("task", obj.get("task_name", obj.get("language_instruction", "unknown"))))
                # Map language instruction to slug
                task_slug = LANGUAGE_TO_SLUG.get(task_lang, f"unknown-{tid}")
                task_map[tid] = task_slug
        return task_map

    def _episode_iter_from_meta(self) -> List[LeRobotEpisodeRef]:
        # Try patched format first (episodes_patched.jsonl), then fall back to original
        patched_path = self.meta_dir / "episodes_patched.jsonl"
        path = patched_path if patched_path.exists() else self.meta_dir / "episodes.jsonl"
        if not path.exists():
            return []
        
        # Load task_index -> slug mapping from tasks.jsonl
        task_idx_to_slug = {}
        with self.meta_dir.joinpath("tasks.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                tid = int(obj.get("task_index", -1))
                task_lang = obj.get("task", "")
                task_idx_to_slug[tid] = task_lang
        
        # Build language -> task_index mapping
        lang_to_idx = {lang: idx for idx, lang in task_idx_to_slug.items()}
        
        refs: List[LeRobotEpisodeRef] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                episode_id = int(obj.get("episode_id", obj.get("id", obj.get("episode_index", -1))))
                
                # Get task_index: check both patched format and original format
                task_idx = obj.get("task_index", -1)
                if task_idx == -1:
                    # Original format: task_index is in tasks.jsonl, not in episodes.jsonl
                    # We need to look up by language instruction
                    tasks_list = obj.get("tasks", [])
                    if tasks_list:
                        lang = tasks_list[0]
                        task_idx = lang_to_idx.get(lang, -1)
                
                # Get task_name using task_map (language description)
                task_name = self.task_map.get(int(task_idx), "unknown")
                
                # Get chunk name
                chunk = str(obj.get("chunk", "chunk-000"))
                if chunk == "chunk-000" and "episode_index" in obj:
                    # Compute chunk from episode_index (1000 episodes per chunk)
                    ep_idx = obj.get("episode_index", 0)
                    chunk_num = ep_idx // 1000
                    chunk = f"chunk-{chunk_num:03d}"
                
                parquet = self.data_dir / chunk / f"episode_{episode_id:06d}.parquet"
                if not parquet.exists():
                    if "parquet_path" in obj:
                        parquet = self.root / str(obj["parquet_path"])
                if not parquet.exists():
                    continue
                length = int(obj.get("length", obj.get("num_steps", -1)))
                if length <= 0:
                    length = -1
                refs.append(
                    LeRobotEpisodeRef(
                        episode_id=episode_id,
                        task_name=task_name,
                        parquet_path=parquet,
                        chunk_name=chunk,
                        num_steps=length,
                    )
                )
        return refs

    def _scan_episodes(self) -> List[LeRobotEpisodeRef]:
        refs = self._episode_iter_from_meta()
        if not refs:
            # fallback: scan parquet files directly
            refs = []
            for p in sorted(self.data_dir.glob("chunk-*/*.parquet")):
                name = p.stem
                try:
                    episode_id = int(name.split("_")[-1])
                except ValueError:
                    continue
                chunk = p.parent.name
                refs.append(
                    LeRobotEpisodeRef(
                        episode_id=episode_id,
                        task_name="unknown",
                        parquet_path=p,
                        chunk_name=chunk,
                        num_steps=-1,
                    )
                )

        if self.cfg.hard_tasks:
            refs = [r for r in refs if r.task_name in self.cfg.hard_tasks]
        if self.cfg.max_episodes is not None:
            refs = refs[: self.cfg.max_episodes]
        if not refs:
            raise RuntimeError("No episodes found for LeRobot MetaWorld dataset.")

        # fill unknown lengths
        filled: List[LeRobotEpisodeRef] = []
        for r in refs:
            if r.num_steps > 0:
                filled.append(r)
                continue
            df = pd.read_parquet(r.parquet_path, columns=[self.cfg.action_key])
            filled.append(
                LeRobotEpisodeRef(
                    episode_id=r.episode_id,
                    task_name=r.task_name,
                    parquet_path=r.parquet_path,
                    chunk_name=r.chunk_name,
                    num_steps=len(df),
                )
            )
        refs = filled

        # split by episode
        rng = np.random.default_rng(self.cfg.seed)
        order = np.arange(len(refs))
        rng.shuffle(order)
        split = int((1.0 - self.cfg.val_ratio) * len(order))
        keep = set(order[:split].tolist()) if self.cfg.split == "train" else set(order[split:].tolist())
        refs = [r for i, r in enumerate(refs) if i in keep]
        if not refs:
            raise RuntimeError(f"No episodes in split={self.cfg.split}")
        return refs

    def _build_index(self) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        t_min = max(
            self.seq_cfg.frame_history - 1,
            self.seq_cfg.state_history - 1,
            self.seq_cfg.action_history - 1,
        )
        h = self.seq_cfg.action_horizon
        for epi_id, ep in enumerate(self.episodes):
            t_max = ep.num_steps - h - 1
            if t_max < t_min:
                continue
            for t in range(t_min, t_max + 1):
                out.append((epi_id, t))
        if not out:
            raise RuntimeError("No valid training samples from LeRobot dataset.")
        return out

    def __len__(self) -> int:
        return len(self.index)

    def _video_path(self, ep: LeRobotEpisodeRef) -> Path:
        # e.g., videos/chunk-000/observation.images.image/episode_000001.mp4
        return self.video_dir / ep.chunk_name / self.cfg.frame_key / f"episode_{ep.episode_id:06d}.mp4"

    def _decode_video(self, path: Path) -> Tensor:
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        import av
        container = av.open(str(path))
        stream = container.streams.video[0]
        frames_list = []
        for frame in container.decode(stream):
            img = frame.to_ndarray(format="rgb24")
            frames_list.append(torch.from_numpy(img))
        container.close()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        frames = torch.stack(frames_list).to(device) if frames_list else torch.zeros(0, 224, 224, 3, dtype=torch.float32, device=device)
        frames = frames.float() / 255.0
        return frames

    def _load_episode_arrays(self, epi_idx: int) -> Dict[str, Tensor]:
        if epi_idx in self._episode_cache:
            return self._episode_cache[epi_idx]
        ep = self.episodes[epi_idx]
        df = pd.read_parquet(ep.parquet_path)

        if self.cfg.states_key not in df.columns or self.cfg.action_key not in df.columns:
            raise KeyError(f"Missing required columns in {ep.parquet_path}")

        states = torch.tensor(np.stack(df[self.cfg.states_key].to_list()), dtype=torch.float32)
        actions = torch.tensor(np.stack(df[self.cfg.action_key].to_list()), dtype=torch.float32)

        if self.cfg.use_stats_norm:
            states = _norm_states(states, self.stats, key=self.cfg.states_key)
            actions = _norm_actions(actions, self.stats, key=self.cfg.action_key)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        arr = {"states": states.to(device), "actions": actions.to(device)}
        if self.cfg.decode_video:
            arr["images"] = self._decode_video(self._video_path(ep))

        self._episode_cache[epi_idx] = arr
        self._episode_order.append(epi_idx)
        while len(self._episode_order) > self.cfg.episode_cache_size:
            old = self._episode_order.pop(0)
            self._episode_cache.pop(old, None)
        return arr

    def _load_vl_npy(self, epi_idx: int) -> Tensor:
        if epi_idx in self._vl_cache:
            cached = self._vl_cache[epi_idx]
            return cached  # None means no features available
        ep = self.episodes[epi_idx]
        npy_path = self._vl_dir / f"episode_{ep.episode_id:06d}.npy"
        if not npy_path.exists():
            self._vl_cache[epi_idx] = None
            return None
        arr = torch.from_numpy(np.load(npy_path)).float()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        arr = arr.to(device)
        self._vl_cache[epi_idx] = arr
        return arr

    def _load_vl_features(self, epi_idx: int, t: int, stride: int = 9) -> Tensor:
        features = self._load_vl_npy(epi_idx)  # [N, 4, 2048]
        if features is None:
            return None
        # Each row corresponds to frame (row_idx * stride). Find nearest to t.
        row = min(t // stride, features.shape[0] - 1)
        return features[row]  # [4, 2048]

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        epi_idx, t = self.index[idx]
        arrays = self._load_episode_arrays(epi_idx)

        states = arrays["states"]
        actions = arrays["actions"]
        if "images" not in arrays:
            raise RuntimeError("decode_video=False is not supported yet for feature construction.")
        images = arrays["images"]

        h_img = self.seq_cfg.frame_history
        h_s = self.seq_cfg.state_history
        h_a = self.seq_cfg.action_history
        h = self.seq_cfg.action_horizon

        frames_hist = images[t - h_img + 1 : t + 1]
        frame_cur = images[t : t + 1]
        future_frames = images[t + 1 : t + 1 + h]

        jepa_tokens = self.featureizer.build_jepa_tokens(frames_hist)
        dino_tokens = self.featureizer.build_dino_tokens(frame_cur)[0]
        vl_features = self._load_vl_features(epi_idx, t)
        if vl_features is None:
            vl_features = self.featureizer.build_vl_features(images, t=t)
        target_future = self.featureizer.build_latent_target(self.featureizer.build_jepa_tokens(future_frames))

        state_hist = states[t - h_s + 1 : t + 1]
        action_hist = actions[t - h_a + 1 : t + 1]
        target_actions = actions[t + 1 : t + 1 + h]

        return {
            "jepa_visual": jepa_tokens,
            "state_hist": state_hist,
            "action_hist": action_hist,
            "dino_current": dino_tokens,
            "vl_features": vl_features,
            "target_actions": target_actions,
            "target_future_latent": target_future,
        }
