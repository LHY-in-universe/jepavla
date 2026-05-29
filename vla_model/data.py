from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class SequenceConfig:
    jepa_window_frames: int = 30
    jepa_refresh_frames: int = 10
    vl_refresh_frames: int = 90
    actions_per_frame: int = 2
    action_horizon: int = 30
    robot_hist_frame_offsets: tuple[int, int, int, int] = (30, 20, 10, 1)


def pad_frame_indices(current_t: int, offsets: Sequence[int]) -> list[int]:
    return [max(0, current_t - int(offset)) for offset in offsets]


def action_slice_from_frame_t(current_t: int, horizon: int, actions_per_frame: int) -> slice:
    start = (current_t + 1) * actions_per_frame
    end = start + horizon
    return slice(start, end)


def is_jepa_tick(frame_idx: int, refresh_frames: int) -> bool:
    return frame_idx >= 0 and (frame_idx + 1) % refresh_frames == 0


def collate_samples(samples: list[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    keys = samples[0].keys()
    out: Dict[str, Tensor] = {}
    for k in keys:
        first = samples[0][k]
        if torch.is_tensor(first):
            out[k] = torch.stack([s[k] for s in samples], dim=0)
        else:
            raise TypeError(f"Unsupported sample value for key `{k}`: {type(first)!r}")
    return out


def valid_tick_indices(
    num_frames: int,
    num_actions: int,
    cfg: SequenceConfig,
) -> Iterable[int]:
    min_frame = cfg.jepa_window_frames - 1
    action_frames_needed = (cfg.action_horizon + cfg.actions_per_frame - 1) // cfg.actions_per_frame
    max_frame = min(num_frames - 1, (num_actions // cfg.actions_per_frame) - action_frames_needed - 1)
    for t in range(min_frame, max_frame + 1):
        if is_jepa_tick(t, cfg.jepa_refresh_frames):
            yield t
