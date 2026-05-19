from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .schema import VLAConfig
from .data import SequenceConfig


def _as_float_tensor(x: np.ndarray) -> Tensor:
    if x.dtype == np.uint8:
        t = torch.from_numpy(x).float() / 255.0
    else:
        t = torch.from_numpy(x).float()
    return t


@dataclass
class EpisodeMeta:
    path: Path
    task_name: str
    num_steps: int


@dataclass
class DatasetConfig:
    root_dir: str
    split: str = "train"  # train | val
    val_ratio: float = 0.1
    seed: int = 42
    hard_tasks: Optional[List[str]] = None
    max_episodes: Optional[int] = None
    require_precomputed_features: bool = False
    episode_cache_size: int = 8


class RandomProjectionFeatureizer:
    """
    Fallback featureizer for raw image episodes when precomputed tokens do not exist.
    This keeps the training pipeline runnable and deterministic.
    """

    def __init__(self, cfg: VLAConfig, patch_size: int = 16, vl_frames: int = 8, seed: int = 42, device: str = "cuda") -> None:
        self.cfg = cfg
        self.patch_size = patch_size
        self.vl_frames = vl_frames
        import torch as _torch
        self._device = device if _torch.cuda.is_available() else "cpu"

        patch_dim = 3 * patch_size * patch_size
        g = torch.Generator()
        g.manual_seed(seed)
        self.proj_jepa = (torch.randn(patch_dim, cfg.jepa_in_dim, generator=g) / math.sqrt(patch_dim)).to(self._device)
        self.proj_dino = (torch.randn(patch_dim, cfg.dino_in_dim, generator=g) / math.sqrt(patch_dim)).to(self._device)
        self.proj_vl = (torch.randn(patch_dim, cfg.vl_in_dim, generator=g) / math.sqrt(patch_dim)).to(self._device)
        self.proj_latent_target = (torch.randn(cfg.jepa_in_dim, cfg.model_dim, generator=g) / math.sqrt(cfg.jepa_in_dim)).to(self._device)

    def _patchify(self, frames: Tensor) -> Tensor:
        # frames: [T, H, W, 3], float in [0,1]
        x = frames.permute(0, 3, 1, 2).contiguous()
        unfold = torch.nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        p = unfold(x).transpose(1, 2).contiguous()
        return p

    def build_jepa_tokens(self, frames_hist: Tensor) -> Tensor:
        patches = self._patchify(frames_hist)
        return (patches @ self.proj_jepa.to(dtype=patches.dtype))

    def build_dino_tokens(self, frame_cur: Tensor) -> Tensor:
        patches = self._patchify(frame_cur)
        return (patches @ self.proj_dino.to(dtype=patches.dtype))

    def build_vl_features(self, frames: Tensor, t: int) -> Tensor:
        start = max(0, t - 31)
        end = t + 1
        idx = torch.linspace(start, end - 1, steps=self.vl_frames).long().clamp(min=0, max=frames.shape[0] - 1)
        sel = frames[idx]
        patches = self._patchify(sel)
        pooled = patches.mean(dim=1)
        return (pooled @ self.proj_vl.to(dtype=pooled.dtype))

    def build_latent_target(self, future_jepa_tokens: Tensor) -> Tensor:
        # future_jepa_tokens: [H, P, Cjepa]
        pooled = future_jepa_tokens.mean(dim=1)
        return (pooled @ self.proj_latent_target.to(dtype=pooled.dtype))


class MetaWorldVLADataset(Dataset):
    """
    Offline dataset for MetaWorld-style trajectories.

    Expected per-episode npz keys:
    - required: states [T, S], actions [T, A]
    - optional:
      - images [T, H, W, 3]
      - task_name (scalar string)
      - jepa_visual_tokens [T, P, Cj]
      - dino_tokens [T, Pd, Cd]
      - vl_features [T, Lv, Cv] or [Lv, Cv]
      - target_future_latent [T, H, D]
    """

    def __init__(self, cfg: DatasetConfig, model_cfg: VLAConfig, seq_cfg: SequenceConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.seq_cfg = seq_cfg
        self.featureizer = RandomProjectionFeatureizer(model_cfg, seed=cfg.seed)

        self.episodes = self._scan_episodes()
        self.index = self._build_index()
        self._cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._cache_order: List[int] = []

    def _scan_episodes(self) -> List[EpisodeMeta]:
        root = Path(self.cfg.root_dir)
        paths = sorted(root.rglob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"No .npz episodes found under {root}")
        if self.cfg.max_episodes is not None:
            paths = paths[: self.cfg.max_episodes]

        metas: List[EpisodeMeta] = []
        for path in paths:
            with np.load(path, allow_pickle=True) as data:
                if "states" not in data or "actions" not in data:
                    continue
                states = data["states"]
                num_steps = int(states.shape[0])
                task_name = "unknown"
                if "task_name" in data:
                    raw = data["task_name"]
                    if np.ndim(raw) == 0:
                        task_name = str(raw.item())
                    else:
                        task_name = str(raw[0])
                if self.cfg.hard_tasks and task_name not in self.cfg.hard_tasks:
                    continue
                metas.append(EpisodeMeta(path=path, task_name=task_name, num_steps=num_steps))
        if not metas:
            raise RuntimeError("No valid episodes after filtering.")

        # deterministic split by episode
        rng = np.random.default_rng(self.cfg.seed)
        order = np.arange(len(metas))
        rng.shuffle(order)
        split = int((1.0 - self.cfg.val_ratio) * len(order))
        if self.cfg.split == "train":
            chosen = set(order[:split].tolist())
        else:
            chosen = set(order[split:].tolist())
        metas = [m for i, m in enumerate(metas) if i in chosen]
        if not metas:
            raise RuntimeError(f"No episodes in split={self.cfg.split}.")
        return metas

    def _build_index(self) -> List[tuple[int, int]]:
        idx: List[tuple[int, int]] = []
        t_min = max(
            self.seq_cfg.frame_history - 1,
            self.seq_cfg.state_history - 1,
            self.seq_cfg.action_history - 1,
        )
        h = self.seq_cfg.action_horizon
        for epi_id, epi in enumerate(self.episodes):
            t_max = epi.num_steps - h - 1
            if t_max < t_min:
                continue
            for t in range(t_min, t_max + 1):
                idx.append((epi_id, t))
        if not idx:
            raise RuntimeError("No valid samples. Check sequence lengths and episode lengths.")
        return idx

    def __len__(self) -> int:
        return len(self.index)

    def _load_episode(self, epi_id: int) -> Dict[str, np.ndarray]:
        if epi_id in self._cache:
            return self._cache[epi_id]
        path = self.episodes[epi_id].path
        with np.load(path, allow_pickle=True) as data:
            episode = {k: data[k] for k in data.files}
        self._cache[epi_id] = episode
        self._cache_order.append(epi_id)
        while len(self._cache_order) > self.cfg.episode_cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        return episode

    def _build_from_precomputed(self, ep: Dict[str, np.ndarray], t: int) -> Optional[Dict[str, Tensor]]:
        if "jepa_visual_tokens" not in ep:
            return None
        h_img = self.seq_cfg.frame_history
        h_s = self.seq_cfg.state_history
        h_a = self.seq_cfg.action_history
        h = self.seq_cfg.action_horizon

        jepa = torch.from_numpy(ep["jepa_visual_tokens"][t - h_img + 1 : t + 1]).float()
        states = torch.from_numpy(ep["states"][t - h_s + 1 : t + 1]).float()
        acts = torch.from_numpy(ep["actions"][t - h_a + 1 : t + 1]).float()
        target_actions = torch.from_numpy(ep["actions"][t + 1 : t + 1 + h]).float()

        if "dino_tokens" in ep:
            dino = torch.from_numpy(ep["dino_tokens"][t]).float()
        else:
            return None

        if "vl_features" in ep:
            raw_vl = ep["vl_features"]
            if raw_vl.ndim == 3:
                vl = torch.from_numpy(raw_vl[t]).float()
            else:
                vl = torch.from_numpy(raw_vl).float()
        else:
            return None

        out: Dict[str, Tensor] = {
            "jepa_visual": jepa,
            "state_hist": states,
            "action_hist": acts,
            "dino_current": dino,
            "vl_features": vl,
            "target_actions": target_actions,
        }
        if "target_future_latent" in ep:
            out["target_future_latent"] = torch.from_numpy(ep["target_future_latent"][t]).float()
        else:
            h = self.seq_cfg.action_horizon
            future_jepa = torch.from_numpy(ep["jepa_visual_tokens"][t + 1 : t + 1 + h]).float()
            out["target_future_latent"] = self.featureizer.build_latent_target(future_jepa)
        return out

    def _build_from_raw(self, ep: Dict[str, np.ndarray], t: int) -> Dict[str, Tensor]:
        if "images" not in ep:
            raise RuntimeError("Episode has no precomputed features and no `images` key.")

        h_img = self.seq_cfg.frame_history
        h_s = self.seq_cfg.state_history
        h_a = self.seq_cfg.action_history
        h = self.seq_cfg.action_horizon

        images = _as_float_tensor(ep["images"])
        states = _as_float_tensor(ep["states"])
        actions = _as_float_tensor(ep["actions"])

        frames_hist = images[t - h_img + 1 : t + 1]
        frame_cur = images[t : t + 1]
        future_frames = images[t + 1 : t + 1 + h]

        jepa_tokens = self.featureizer.build_jepa_tokens(frames_hist)
        dino_tokens = self.featureizer.build_dino_tokens(frame_cur)[0]
        vl_features = self.featureizer.build_vl_features(images, t=t)

        states_hist = states[t - h_s + 1 : t + 1]
        action_hist = actions[t - h_a + 1 : t + 1]
        target_actions = actions[t + 1 : t + 1 + h]

        future_jepa_tokens = self.featureizer.build_jepa_tokens(future_frames)
        latent_target = self.featureizer.build_latent_target(future_jepa_tokens)

        return {
            "jepa_visual": jepa_tokens,
            "state_hist": states_hist,
            "action_hist": action_hist,
            "dino_current": dino_tokens,
            "vl_features": vl_features,
            "target_actions": target_actions,
            "target_future_latent": latent_target,
        }

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        epi_id, t = self.index[idx]
        ep = self._load_episode(epi_id)

        sample = self._build_from_precomputed(ep, t)
        if sample is None:
            if self.cfg.require_precomputed_features:
                raise RuntimeError(
                    f"Episode {self.episodes[epi_id].path} missing required precomputed features."
                )
            sample = self._build_from_raw(ep, t)
        return sample
