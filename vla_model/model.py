from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn

from .schema import VLAConfig
from .flow_action_head import FlowActionHead, FlowMatchOutput


def _cross_pool(queries: Tensor, memory: Tensor, attn: nn.MultiheadAttention) -> Tensor:
    batch_size = memory.shape[0]
    q = queries[None].expand(batch_size, -1, -1)
    pooled, _ = attn(q, memory, memory, need_weights=False)
    return pooled


class FiLMPredictorBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
        # FiLM is applied in predictor only (not encoder), following the chosen design.
        h = self.norm1(x)
        h = (1.0 + gamma[:, None, :]) * h + beta[:, None, :]
        x = x + self.dropout(self.attn(h, h, h, need_weights=False)[0])
        h2 = self.norm2(x)
        h2 = (1.0 + gamma[:, None, :]) * h2 + beta[:, None, :]
        x = x + self.dropout(self.ff(h2))
        return x


@dataclass
class ModelOutput:
    cond_tokens: Tensor
    aq_tokens: Tensor
    z_jepa: Tensor
    z_dino: Tensor
    pred_future_latent: Tensor
    sampled_actions: Optional[Tensor] = None
    flow: Optional[FlowMatchOutput] = None


class ThinkJEPAVLAModel(nn.Module):
    """
    Prototype wiring for:
    VL-guided JEPA + JEPA2AC || DINO + Flow Matching Action Head + RTC-ready API.

    Expected batch keys:
    - jepa_visual: [B, T, P, jepa_in_dim]
    - state_hist: [B, T, jepa_state_dim]
    - action_hist: [B, T, action_dim]
    - dino_current: [B, P_d, dino_in_dim]
    - vl_features: [B, L_vl, vl_in_dim]
    Optional:
    - target_actions: [B, H, action_dim]
    - target_future_latent: [B, H, model_dim]
    """

    def __init__(self, config: VLAConfig) -> None:
        super().__init__()
        self.config = config
        d = config.model_dim

        self.jepa_visual_proj = nn.Linear(config.jepa_in_dim, d)
        self.state_proj = nn.Linear(config.jepa_state_dim, d)
        self.action_hist_proj = nn.Linear(config.action_dim, d)
        self.use_dino = getattr(config, "use_dino", True)
        self.vl_proj = nn.Linear(config.vl_in_dim, d)
        self.dino_proj = nn.Linear(config.dino_in_dim, d)
        self.vl_dropout = nn.Dropout(config.vl_guidance_dropout)

        self.predictor_blocks = nn.ModuleList(
            [
                FiLMPredictorBlock(dim=d, num_heads=config.num_heads, dropout=config.dropout)
                for _ in range(config.jepa_predictor_layers)
            ]
        )
        self.film_mlps = nn.ModuleList([nn.Linear(d, 2 * d) for _ in range(config.jepa_predictor_layers)])
        self.predictor_out_norm = nn.LayerNorm(d)

        self.future_queries = nn.Parameter(torch.randn(config.action_horizon, d) * (d**-0.5))
        self.future_decoder_attn = nn.MultiheadAttention(d, config.num_heads, batch_first=True)
        self.future_out = nn.Linear(d, d)

        self.jepa_action_queries = nn.Parameter(torch.randn(config.num_jepa_action_tokens, d) * (d**-0.5))
        self.jepa_action_attn = nn.MultiheadAttention(d, config.num_heads, batch_first=True)

        self.dino_queries = nn.Parameter(torch.randn(config.num_dino_tokens, d) * (d**-0.5))
        self.dino_attn = nn.MultiheadAttention(d, config.num_heads, batch_first=True)

        self.special_bos = nn.Parameter(torch.randn(1, 1, d) * (d**-0.5))
        self.special_jsep = nn.Parameter(torch.randn(1, 1, d) * (d**-0.5))
        self.special_dsep = nn.Parameter(torch.randn(1, 1, d) * (d**-0.5))
        self.special_ssep = nn.Parameter(torch.randn(1, 1, d) * (d**-0.5))
        self.action_query_tokens = nn.Parameter(torch.randn(config.action_horizon, d) * (d**-0.5))
        self.segment_embed = nn.Embedding(5, d)
        self.cond_out_norm = nn.LayerNorm(d)
        self.cond_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d,
                nhead=config.num_heads,
                dim_feedforward=4 * d,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=2,
        )

        self.flow_head = FlowActionHead(
            model_dim=d,
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            num_heads=config.num_heads,
            num_layers=config.flow_num_layers,
            time_embed_dim=config.flow_time_embed_dim,
            dropout=config.dropout,
            solver_steps=config.flow_solver_steps,
        )

    def _build_jepa_tokens(self, jepa_visual: Tensor, state_hist: Tensor, action_hist: Tensor) -> Tensor:
        x_vis = self.jepa_visual_proj(jepa_visual)  # [B, T, P, D]
        t_vis = x_vis.shape[1]
        # Align robot histories to visual context length (use most recent slice).
        if state_hist.shape[1] != t_vis:
            state_hist = state_hist[:, -t_vis:, :]
        if action_hist.shape[1] != t_vis:
            action_hist = action_hist[:, -t_vis:, :]
        x_state = self.state_proj(state_hist)[:, :, None, :]  # [B, T, 1, D]
        x_action = self.action_hist_proj(action_hist)[:, :, None, :]  # [B, T, 1, D]
        x = torch.cat([x_action, x_state, x_vis], dim=2)  # [B, T, 2+P, D]
        b, t, n, d = x.shape
        x = x.reshape(b, t * n, d)
        return x

    def _predict_future_latent(self, context_tokens: Tensor) -> Tensor:
        q = self.future_queries[None].expand(context_tokens.shape[0], -1, -1)
        q2, _ = self.future_decoder_attn(q, context_tokens, context_tokens, need_weights=False)
        return self.future_out(q2)

    def _vl_guided_predictor(self, context_tokens: Tensor, vl_features: Tensor) -> Tensor:
        g_tokens = self.vl_proj(vl_features)
        g = self.vl_dropout(g_tokens.mean(dim=1))
        x = context_tokens
        for block, film_mlp in zip(self.predictor_blocks, self.film_mlps):
            gamma, beta = film_mlp(g).chunk(2, dim=-1)
            x = block(x, gamma=gamma, beta=beta)
        return self.predictor_out_norm(x)

    def _build_cond_tokens(self, z_jepa: Tensor, z_dino: Tensor, state_token: Tensor) -> tuple[Tensor, Tensor]:
        bsz = z_jepa.shape[0]
        bos = self.special_bos.expand(bsz, -1, -1)
        jsep = self.special_jsep.expand(bsz, -1, -1)
        dsep = self.special_dsep.expand(bsz, -1, -1)
        ssep = self.special_ssep.expand(bsz, -1, -1)
        aq = self.action_query_tokens[None].expand(bsz, -1, -1)

        if z_dino.shape[1] > 0:
            cond = torch.cat([bos, jsep, z_jepa, dsep, z_dino, ssep, state_token, aq], dim=1)
            seg_ids = torch.cat([
                torch.zeros(2, device=cond.device, dtype=torch.long),       # bos, jsep
                torch.ones(z_jepa.shape[1], device=cond.device, dtype=torch.long),
                torch.zeros(1, device=cond.device, dtype=torch.long),       # dsep
                torch.full((z_dino.shape[1],), 2, device=cond.device, dtype=torch.long),
                torch.zeros(1, device=cond.device, dtype=torch.long),       # ssep
                torch.full((state_token.shape[1],), 3, device=cond.device, dtype=torch.long),
                torch.full((aq.shape[1],), 4, device=cond.device, dtype=torch.long),
            ])
        else:
            cond = torch.cat([bos, jsep, z_jepa, ssep, state_token, aq], dim=1)
            seg_ids = torch.cat([
                torch.zeros(2, device=cond.device, dtype=torch.long),       # bos, jsep
                torch.ones(z_jepa.shape[1], device=cond.device, dtype=torch.long),
                torch.zeros(1, device=cond.device, dtype=torch.long),       # ssep
                torch.full((state_token.shape[1],), 3, device=cond.device, dtype=torch.long),
                torch.full((aq.shape[1],), 4, device=cond.device, dtype=torch.long),
            ])
        cond = cond + self.segment_embed(seg_ids)[None]
        cond = self.cond_encoder(cond)
        cond = self.cond_out_norm(cond)
        aq_start = cond.shape[1] - self.config.action_horizon
        aq_tokens = cond[:, aq_start:, :]
        return cond, aq_tokens

    def forward(
        self,
        batch: Dict[str, Tensor],
        sample_actions: bool = False,
        num_flow_steps: Optional[int] = None,
    ) -> ModelOutput:
        jepa_tokens = self._build_jepa_tokens(
            jepa_visual=batch["jepa_visual"],
            state_hist=batch["state_hist"],
            action_hist=batch["action_hist"],
        )

        guided_tokens = self._vl_guided_predictor(context_tokens=jepa_tokens, vl_features=batch["vl_features"])
        pred_future_latent = self._predict_future_latent(guided_tokens)

        jepa_memory = torch.cat([guided_tokens, pred_future_latent], dim=1)
        z_jepa = _cross_pool(self.jepa_action_queries, jepa_memory, self.jepa_action_attn)

        if self.use_dino:
            dino_tokens = self.dino_proj(batch["dino_current"])
            z_dino = _cross_pool(self.dino_queries, dino_tokens, self.dino_attn)
        else:
            d_model = self.config.model_dim
            z_dino = torch.zeros(jepa_tokens.shape[0], 0, d_model, device=jepa_tokens.device, dtype=jepa_tokens.dtype)

        state_token = self.state_proj(batch["state_hist"][:, -1:, :])
        cond_tokens, aq_tokens = self._build_cond_tokens(z_jepa=z_jepa, z_dino=z_dino, state_token=state_token)

        sampled = None
        if sample_actions:
            sampled = self.flow_head.sample(cond_tokens=cond_tokens, aq_tokens=aq_tokens, num_steps=num_flow_steps)

        flow = None
        if "target_actions" in batch:
            flow = self.flow_head.training_loss(
                cond_tokens=cond_tokens,
                target_actions=batch["target_actions"],
                aq_tokens=aq_tokens,
            )

        return ModelOutput(
            cond_tokens=cond_tokens,
            aq_tokens=aq_tokens,
            z_jepa=z_jepa,
            z_dino=z_dino,
            pred_future_latent=pred_future_latent,
            sampled_actions=sampled,
            flow=flow,
        )

    @torch.no_grad()
    def predict_action_chunk(self, batch: Dict[str, Tensor], num_flow_steps: Optional[int] = None) -> Tensor:
        out = self.forward(batch, sample_actions=True, num_flow_steps=num_flow_steps)
        if out.sampled_actions is None:
            raise RuntimeError("No sampled actions were produced.")
        return out.sampled_actions


def compute_vla_loss(output: ModelOutput, batch: Dict[str, Tensor], config: VLAConfig) -> Dict[str, Tensor]:
    losses: Dict[str, Tensor] = {}

    if output.flow is None:
        raise ValueError("Flow output is missing. Provide `target_actions` in batch.")

    losses["flow"] = output.flow.loss
    total = output.flow.loss

    if "target_future_latent" in batch:
        latent = nn.functional.mse_loss(output.pred_future_latent, batch["target_future_latent"])
        losses["jepa_latent"] = latent
        total = total + config.lambda_jepa_latent * latent
    else:
        losses["jepa_latent"] = torch.zeros((), device=total.device, dtype=total.dtype)

    losses["total"] = total
    return losses
