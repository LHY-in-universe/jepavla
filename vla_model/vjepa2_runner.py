from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .schema import VJEPA2Config


@dataclass
class VJEPA2Output:
    context_tokens: Tensor
    target_tokens: Tensor
    pooled_context: Tensor


class NativeVJEPA2Model(nn.Module):
    def __init__(self, cfg: VJEPA2Config) -> None:
        super().__init__()
        from transformers import VJEPA2Model as HFVJEPA2

        self.cfg = cfg
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.dtype, torch.float32)
        self.model = HFVJEPA2.from_pretrained(
            "facebook/vjepa2-vitl-fpc64-256",
            torch_dtype=dtype,
            local_files_only=True,
        )
        self.model.eval()

    def forward(self, frames: Tensor) -> VJEPA2Output:
        # frames: [B, T, H, W, 3] float32 in [0, 1]
        B, T, H, W, C = frames.shape

        # Resize to model's expected image size (256)
        if H != self.cfg.image_size or W != self.cfg.image_size:
            frames_rs = F.interpolate(
                frames.permute(0, 4, 1, 2, 3).float(),
                size=(T, self.cfg.image_size, self.cfg.image_size),
                mode="trilinear",
                align_corners=False,
            )
            # [B, 3, T, H, W] → [B, T, 3, H, W]
            pixel_values = frames_rs.permute(0, 2, 1, 3, 4)
        else:
            pixel_values = frames.permute(0, 3, 1, 2)  # [B, 3, T, H, W]? No, need [B, T, 3, H, W]
            # frames is [B, T, H, W, 3] → permute to [B, T, 3, H, W]
            pixel_values = frames.permute(0, 1, 4, 2, 3).contiguous()

        pixel_values = pixel_values.to(dtype=self.model.dtype)

        with torch.no_grad():
            out = self.model(pixel_values_videos=pixel_values, skip_predictor=True)

        context_tokens = out.last_hidden_state  # [B, num_patches, 1024]
        pooled_context = context_tokens.mean(dim=1)  # [B, 1024]

        # Target tokens are the encoder's own output, detached (no grad into backbone)
        target_tokens = context_tokens.detach()

        return VJEPA2Output(
            context_tokens=context_tokens,
            target_tokens=target_tokens,
            pooled_context=pooled_context,
        )

    def freeze_backbone(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False


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
        return NativeVJEPA2Model(self.cfg)

    def forward(self, context_frames: Tensor) -> VJEPA2Output:
        return self.backend(context_frames)

    def freeze_backbone(self) -> None:
        if isinstance(self.backend, NativeVJEPA2Model):
            self.backend.freeze_backbone()
