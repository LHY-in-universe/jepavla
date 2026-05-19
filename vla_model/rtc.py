from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .model import ThinkJEPAVLAModel


@dataclass
class RTCConfig:
    chunk_horizon: int = 8
    execute_steps: int = 4
    execution_horizon: int = 4
    max_guidance_weight: float = 10.0
    prefix_attention_schedule: str = "exp"  # "exp" or "linear"


class RTCExecutor:
    """
    Real-time chunking wrapper:
    - predicts a new action chunk
    - blends with leftover from previous chunk
    - executes first half-chunk
    """

    def __init__(self, policy: ThinkJEPAVLAModel, config: RTCConfig) -> None:
        self.policy = policy
        self.config = config
        self.leftover_chunk: Optional[Tensor] = None

    def reset(self) -> None:
        self.leftover_chunk = None

    def _guidance_weight(self, i: int) -> float:
        denom = max(self.config.execution_horizon - 1, 1)
        if self.config.prefix_attention_schedule.lower() == "linear":
            alpha = 1.0 - float(i) / float(denom)
        else:
            alpha = torch.exp(torch.tensor(-float(i) / float(denom))).item()
        scale = self.config.max_guidance_weight / (1.0 + self.config.max_guidance_weight)
        return max(0.0, min(1.0, alpha * scale))

    def _merge_with_leftover(self, new_chunk: Tensor, delay_steps: int = 0) -> Tensor:
        if self.leftover_chunk is None:
            return new_chunk

        merged = new_chunk.clone()
        overlap = min(self.leftover_chunk.shape[1], merged.shape[1], self.config.execution_horizon)
        for i in range(overlap):
            if i < delay_steps:
                merged[:, i] = self.leftover_chunk[:, i]
                continue
            w = self._guidance_weight(i)
            merged[:, i] = w * self.leftover_chunk[:, i] + (1.0 - w) * merged[:, i]
        return merged

    @torch.no_grad()
    def step(self, batch: dict, delay_steps: int = 0, num_flow_steps: Optional[int] = None) -> tuple[Tensor, Tensor]:
        """
        Returns:
        - actions_to_execute: [B, execute_steps, A]
        - merged_chunk: [B, H, A]
        """
        chunk = self.policy.predict_action_chunk(batch, num_flow_steps=num_flow_steps)
        chunk = self._merge_with_leftover(chunk, delay_steps=delay_steps)
        execute = chunk[:, : self.config.execute_steps]
        self.leftover_chunk = chunk[:, self.config.execute_steps :]
        return execute, chunk
