from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor


@dataclass
class SequenceConfig:
    frame_history: int = 4  # t-3..t
    state_history: int = 16
    action_history: int = 16
    action_horizon: int = 8


def build_no_leak_sample(
    images: Tensor,
    states: Tensor,
    actions: Tensor,
    t: int,
    cfg: SequenceConfig,
) -> Dict[str, Tensor]:
    """
    Build one no-leak training sample for control time t.

    Inputs:
    - images: [T_total, ...]
    - states: [T_total, S]
    - actions: [T_total, A] where actions[k] is executed between k and k+1

    At control time t:
    - context uses <= t
    - target predicts t+1..t+H
    """
    total = images.shape[0]
    if states.shape[0] != total or actions.shape[0] != total:
        raise ValueError("images/states/actions must share first dimension.")

    h_img = cfg.frame_history
    h_s = cfg.state_history
    h_a = cfg.action_history
    h = cfg.action_horizon

    if t < h_img - 1:
        raise ValueError("t is too small for frame history.")
    if t < h_s - 1 or t < h_a - 1:
        raise ValueError("t is too small for state/action history.")
    if t + h >= total:
        raise ValueError("Not enough future steps for action target.")

    frame_slice = slice(t - h_img + 1, t + 1)
    state_slice = slice(t - h_s + 1, t + 1)
    action_hist_slice = slice(t - h_a + 1, t + 1)
    target_slice = slice(t + 1, t + 1 + h)

    return {
        "frames_jepa": images[frame_slice],        # includes current frame
        "frame_dino": images[t : t + 1],           # current frame only
        "state_hist": states[state_slice],         # includes current state
        "action_hist": actions[action_hist_slice], # includes current executed action
        "target_actions": actions[target_slice],   # strictly future
    }


def collate_samples(samples: list[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    keys = samples[0].keys()
    out: Dict[str, Tensor] = {}
    for k in keys:
        out[k] = torch.stack([s[k] for s in samples], dim=0)
    return out
