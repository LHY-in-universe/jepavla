from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .data import SequenceConfig, action_slice_from_frame_t, pad_frame_indices, valid_tick_indices
from .scheduler import ControlScheduler
from .schema import VLAConfig


def _as_float_tensor(x: np.ndarray) -> Tensor:
    if x.dtype == np.uint8:
        return torch.from_numpy(x).float() / 255.0
    return torch.from_numpy(x).float()


@dataclass
class EpisodeMeta:
    path: Path
    task_name: str
    num_frames: int
    num_actions: int


@dataclass
class DatasetConfig:
    root_dir: str
    split: str = "train"
    val_ratio: float = 0.1
    seed: int = 42
    hard_tasks: Optional[List[str]] = None
    max_episodes: Optional[int] = None
    episode_cache_size: int = 8


class MetaWorldVLADataset(Dataset):
    def __init__(self, cfg: DatasetConfig, model_cfg: VLAConfig, seq_cfg: SequenceConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.seq_cfg = seq_cfg
        self.scheduler = ControlScheduler(model_cfg.scheduler)
        self.episodes = self._scan_episodes()
        self.index = self._build_index()
        self._cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._cache_order: List[int] = []

    def _scan_episodes(self) -> List[EpisodeMeta]:
        root = Path(self.cfg.root_dir)
        paths = sorted(root.rglob("*.npz"))
        if self.cfg.max_episodes is not None:
            paths = paths[: self.cfg.max_episodes]
        metas: List[EpisodeMeta] = []
        for path in paths:
            with np.load(path, allow_pickle=True) as data:
                if "images" not in data or "states" not in data or "actions" not in data:
                    continue
                task_name = "unknown"
                if "task_name" in data:
                    raw = data["task_name"]
                    task_name = str(raw.item()) if np.ndim(raw) == 0 else str(raw[0])
                if self.cfg.hard_tasks and task_name not in self.cfg.hard_tasks:
                    continue
                metas.append(
                    EpisodeMeta(
                        path=path,
                        task_name=task_name,
                        num_frames=int(data["images"].shape[0]),
                        num_actions=int(data["actions"].shape[0]),
                    )
                )
        if not metas:
            raise RuntimeError(f"No valid episodes found under {root}")
        rng = np.random.default_rng(self.cfg.seed)
        order = np.arange(len(metas))
        rng.shuffle(order)
        split = int((1.0 - self.cfg.val_ratio) * len(order))
        keep = set(order[:split].tolist()) if self.cfg.split == "train" else set(order[split:].tolist())
        metas = [m for i, m in enumerate(metas) if i in keep]
        if not metas:
            raise RuntimeError(f"No episodes in split={self.cfg.split}")
        return metas

    def _build_index(self) -> list[tuple[int, int]]:
        idx: list[tuple[int, int]] = []
        for epi_id, epi in enumerate(self.episodes):
            for frame_t in valid_tick_indices(epi.num_frames, epi.num_actions, self.seq_cfg):
                idx.append((epi_id, frame_t))
        if not idx:
            raise RuntimeError("No valid JEPA tick samples in dataset.")
        return idx

    def __len__(self) -> int:
        return len(self.index)

    def _load_episode(self, epi_id: int) -> Dict[str, np.ndarray]:
        if epi_id in self._cache:
            return self._cache[epi_id]
        with np.load(self.episodes[epi_id].path, allow_pickle=True) as data:
            episode = {k: data[k] for k in data.files}
        self._cache[epi_id] = episode
        self._cache_order.append(epi_id)
        while len(self._cache_order) > self.cfg.episode_cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        return episode

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        epi_id, frame_t = self.index[idx]
        ep = self._load_episode(epi_id)
        images = _as_float_tensor(ep["images"])
        states = _as_float_tensor(ep["states"])
        actions = _as_float_tensor(ep["actions"])

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
                f"Episode {self.episodes[epi_id].path} produced invalid target horizon {target_actions.shape[0]}"
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
