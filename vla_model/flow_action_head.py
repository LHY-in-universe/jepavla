from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


def sinusoidal_time_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """Build sinusoidal embeddings for scalar diffusion/flow time."""
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(float(max_period), device=t.device, dtype=t.dtype))
        * torch.arange(half, device=t.device, dtype=t.dtype)
        / max(half, 1)
    )
    angles = t[:, None] * freqs[None]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class FlowCrossBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_q1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_q2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: Tensor, memory: Tensor) -> Tensor:
        q1 = self.norm_q1(q)
        q = q + self.dropout(self.self_attn(q1, q1, q1, need_weights=False)[0])
        q2 = self.norm_q2(q)
        q = q + self.dropout(self.cross_attn(q2, memory, memory, need_weights=False)[0])
        q = q + self.dropout(self.ff(self.norm_ff(q)))
        return q


@dataclass
class FlowMatchOutput:
    loss: Tensor
    predicted_velocity: Tensor
    target_velocity: Tensor
    interpolated_action: Tensor
    tau: Tensor


class FlowActionHead(nn.Module):
    """
    Conditional Flow Matching action head.

    It learns a velocity field v_theta(x_t, t | cond_tokens) and integrates from
    Gaussian noise x_0 to action chunk x_1.
    """

    def __init__(
        self,
        model_dim: int,
        action_dim: int,
        action_horizon: int,
        num_heads: int,
        num_layers: int,
        time_embed_dim: int,
        dropout: float,
        solver_steps: int,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.solver_steps = solver_steps

        self.action_in = nn.Linear(action_dim, model_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.query_embed = nn.Parameter(torch.randn(action_horizon, model_dim) * (model_dim**-0.5))
        self.blocks = nn.ModuleList(
            [FlowCrossBlock(model_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm_out = nn.LayerNorm(model_dim)
        self.velocity_out = nn.Linear(model_dim, action_dim)

        self.time_embed_dim = time_embed_dim

    def forward(self, x_t: Tensor, tau: Tensor, cond_tokens: Tensor, aq_tokens: Optional[Tensor] = None) -> Tensor:
        if tau.ndim == 0:
            tau = tau.repeat(x_t.shape[0])
        if tau.ndim == 2 and tau.shape[-1] == 1:
            tau = tau[:, 0]

        t_embed = sinusoidal_time_embedding(tau.to(x_t.dtype), self.time_embed_dim)
        t_embed = self.time_mlp(t_embed)[:, None, :]

        q = self.action_in(x_t) + t_embed + self.query_embed[None]
        if aq_tokens is not None:
            q = q + aq_tokens

        for block in self.blocks:
            q = block(q, cond_tokens)

        velocity = self.velocity_out(self.norm_out(q))
        return velocity

    def training_loss(self, cond_tokens: Tensor, target_actions: Tensor, aq_tokens: Optional[Tensor] = None) -> FlowMatchOutput:
        batch_size = target_actions.shape[0]
        tau = torch.rand(batch_size, device=target_actions.device, dtype=target_actions.dtype)
        eps = torch.randn_like(target_actions)
        x_t = (1.0 - tau[:, None, None]) * eps + tau[:, None, None] * target_actions
        target_velocity = target_actions - eps
        pred_velocity = self.forward(x_t=x_t, tau=tau, cond_tokens=cond_tokens, aq_tokens=aq_tokens)
        loss = nn.functional.mse_loss(pred_velocity, target_velocity)
        return FlowMatchOutput(
            loss=loss,
            predicted_velocity=pred_velocity,
            target_velocity=target_velocity,
            interpolated_action=x_t,
            tau=tau,
        )

    @torch.no_grad()
    def sample(
        self,
        cond_tokens: Tensor,
        aq_tokens: Optional[Tensor] = None,
        num_steps: Optional[int] = None,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        steps = int(num_steps or self.solver_steps)
        batch_size = cond_tokens.shape[0]
        if noise is None:
            x = torch.randn(
                batch_size,
                self.action_horizon,
                self.action_dim,
                device=cond_tokens.device,
                dtype=cond_tokens.dtype,
            )
        else:
            x = noise

        dt = 1.0 / float(steps)
        for i in range(steps):
            t = torch.full((batch_size,), i / float(steps), device=cond_tokens.device, dtype=cond_tokens.dtype)
            velocity = self.forward(x_t=x, tau=t, cond_tokens=cond_tokens, aq_tokens=aq_tokens)
            x = x + dt * velocity
        return x
