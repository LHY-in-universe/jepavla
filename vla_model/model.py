from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn

from .flow_action_head import FlowActionHead, FlowMatchOutput
from .schema import VLAConfig
from .vjepa2_runner import VJEPA2Output, VJEPA2Runner


class SparseHistoryEncoder(nn.Module):
    def __init__(self, config: VLAConfig) -> None:
        super().__init__()
        d = config.model_dim
        self.image_proj = nn.Linear(3, config.sparse_image_dim)
        self.state_proj = nn.Linear(config.state_dim, config.state_embed_dim)
        self.action_proj = nn.Linear(config.action_dim, config.action_embed_dim)
        self.out_proj = nn.Linear(config.sparse_image_dim + config.state_embed_dim + config.action_embed_dim, d)

    def forward(self, images: Tensor, states: Tensor, actions: Tensor) -> Tensor:
        img_rgb = images.mean(dim=(2, 3))
        img_feat = self.image_proj(img_rgb)
        state_feat = self.state_proj(states)
        action_feat = self.action_proj(actions)
        return self.out_proj(torch.cat([img_feat, state_feat, action_feat], dim=-1))


class PredictorAdapter(nn.Module):
    def __init__(self, config: VLAConfig) -> None:
        super().__init__()
        d = config.model_dim
        self.vl_proj = nn.Linear(config.vl_in_dim, d)
        self.hist_proj = nn.Linear(d, d)
        self.context_proj = nn.Linear(config.jepa_target_dim, d)
        self.target_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, config.jepa_target_dim),
        )
        self.dropout = nn.Dropout(config.vl_guidance_dropout)

    def forward(self, vjepa: VJEPA2Output, hist_tokens: Tensor, vl_cond: Tensor) -> tuple[Tensor, Tensor]:
        hist = self.hist_proj(hist_tokens.mean(dim=1))
        vl = self.dropout(self.vl_proj(vl_cond.mean(dim=1)))
        ctx = self.context_proj(vjepa.pooled_context)
        fused = ctx + hist + vl
        pred_target = self.target_head(fused)
        control_tokens = self.context_proj(vjepa.context_tokens) + fused[:, None, :]
        return pred_target, control_tokens


class JEPAAggregator(nn.Module):
    def __init__(self, config: VLAConfig) -> None:
        super().__init__()
        d = config.model_dim
        self.queries = nn.Parameter(torch.randn(config.z_jepa_tokens, d) * (d**-0.5))
        self.attn = nn.MultiheadAttention(d, config.num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, control_tokens: Tensor, hist_tokens: Tensor) -> Tensor:
        bsz = control_tokens.shape[0]
        q = self.queries[None].expand(bsz, -1, -1)
        memory = torch.cat([control_tokens, hist_tokens], dim=1)
        pooled, _ = self.attn(q, memory, memory, need_weights=False)
        return self.norm(pooled)


@dataclass
class ModelOutput:
    z_jepa: Tensor
    predicted_jepa_targets: Tensor
    target_jepa_targets: Tensor
    aq_tokens: Tensor
    sampled_actions: Optional[Tensor] = None
    flow: Optional[FlowMatchOutput] = None


class ThinkJEPAVLAModel(nn.Module):
    def __init__(self, config: VLAConfig) -> None:
        super().__init__()
        self.config = config
        self.vjepa2 = VJEPA2Runner(config.vjepa2, target_dim=config.jepa_target_dim)
        if config.vjepa2.freeze_backbone:
            self.vjepa2.freeze_backbone()

        self.sparse_hist_encoder = SparseHistoryEncoder(config)
        self.predictor_adapter = PredictorAdapter(config)
        self.jepa_aggregator = JEPAAggregator(config)
        self.state_summary = nn.Linear(config.model_dim, config.model_dim)
        self.action_query_tokens = nn.Parameter(torch.randn(config.action_horizon, config.model_dim) * (config.model_dim**-0.5))
        self.cond_norm = nn.LayerNorm(config.model_dim)
        self.flow_head = FlowActionHead(
            model_dim=config.model_dim,
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            num_heads=config.num_heads,
            num_layers=config.flow_num_layers,
            time_embed_dim=config.flow_time_embed_dim,
            dropout=config.dropout,
            solver_steps=config.flow_solver_steps,
        )

    def _default_vl_cond(self, batch: Dict[str, Tensor]) -> Tensor:
        context = batch["context_frames"]
        bsz = context.shape[0]
        return torch.zeros(bsz, 1, self.config.vl_in_dim, device=context.device, dtype=context.dtype)

    def _build_cond_tokens(self, z_jepa: Tensor, hist_tokens: Tensor) -> tuple[Tensor, Tensor]:
        hist_summary = self.state_summary(hist_tokens.mean(dim=1))
        aq = self.action_query_tokens[None].expand(z_jepa.shape[0], -1, -1)
        cond = torch.cat([z_jepa, hist_summary[:, None, :], aq], dim=1)
        cond = self.cond_norm(cond)
        aq_tokens = cond[:, -self.config.action_horizon :, :]
        return cond, aq_tokens

    def forward(
        self,
        batch: Dict[str, Tensor],
        sample_actions: bool = False,
        num_flow_steps: Optional[int] = None,
    ) -> ModelOutput:
        context_frames = batch["context_frames"]
        sparse_hist_images = batch["sparse_hist_images"]
        sparse_hist_states = batch["sparse_hist_states"]
        sparse_hist_actions = batch["sparse_hist_actions"]
        vl_cond = batch.get("vl_cond")
        if vl_cond is None:
            vl_cond = self._default_vl_cond(batch)

        vjepa = self.vjepa2(context_frames)
        hist_tokens = self.sparse_hist_encoder(sparse_hist_images, sparse_hist_states, sparse_hist_actions)
        predicted_jepa_targets, control_tokens = self.predictor_adapter(vjepa, hist_tokens, vl_cond)
        z_jepa = self.jepa_aggregator(control_tokens, hist_tokens)
        cond_tokens, aq_tokens = self._build_cond_tokens(z_jepa, hist_tokens)

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
            z_jepa=z_jepa,
            predicted_jepa_targets=predicted_jepa_targets,
            target_jepa_targets=vjepa.target_tokens.mean(dim=1),
            aq_tokens=aq_tokens,
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
    if output.flow is None:
        raise ValueError("Flow output is missing. Provide `target_actions` in batch.")

    action_loss = output.flow.loss
    jepa_pred = nn.functional.mse_loss(output.predicted_jepa_targets, output.target_jepa_targets)
    total = config.lambda_action * action_loss + config.lambda_jepa_pred * jepa_pred
    return {
        "total": total,
        "action": action_loss,
        "jepa_pred": jepa_pred,
    }
