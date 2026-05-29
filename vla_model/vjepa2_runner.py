from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn

from .schema import VJEPA2Config


@dataclass
class VJEPA2Output:
    context_tokens: Tensor
    target_tokens: Tensor
    pooled_context: Tensor


class MockVJEPA2Model(nn.Module):
    def __init__(self, target_dim: int) -> None:
        super().__init__()
        self.target_dim = target_dim
        self.proj = nn.Linear(3, target_dim)

    def forward(self, frames: Tensor) -> VJEPA2Output:
        pooled = frames.mean(dim=(1, 2, 3))
        pooled_rgb = frames.mean(dim=(1, 2, 3))
        rgb = frames.mean(dim=(1, 2))
        context_tokens = self.proj(rgb)
        pooled_context = context_tokens.mean(dim=1)
        return VJEPA2Output(
            context_tokens=context_tokens,
            target_tokens=context_tokens.detach(),
            pooled_context=pooled_context,
        )


class VJEPA2Runner(nn.Module):
    def __init__(self, cfg: VJEPA2Config, target_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.target_dim = target_dim
        self.backend = self._build_backend()

    def _build_backend(self) -> nn.Module:
        if self.cfg.allow_mock_runner:
            return MockVJEPA2Model(self.target_dim)
        if not self.cfg.model_name_or_path:
            raise RuntimeError(
                "VJEPA2 model path is not configured. Set model.vjepa2.model_name_or_path "
                "or enable model.vjepa2.allow_mock_runner for development."
            )
        raise RuntimeError(
            "Native VJEPA2 loading is not yet available in this workspace. "
            "Provide a local integration or enable allow_mock_runner for development."
        )

    def forward(self, context_frames: Tensor) -> VJEPA2Output:
        return self.backend(context_frames)

    def freeze_backbone(self) -> None:
        for p in self.backend.parameters():
            p.requires_grad = False
