from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
from torch import nn

from .schema import VLAConfig
from .model import ThinkJEPAVLAModel


@dataclass
class StageHyperParams:
    name: str
    steps: int
    lr_main: float
    lr_backbone: float
    weight_decay: float = 0.05


def default_stage_hparams() -> Dict[str, StageHyperParams]:
    return {
        "A": StageHyperParams(name="A", steps=250_000, lr_main=2e-4, lr_backbone=5e-5),
        "B": StageHyperParams(name="B", steps=200_000, lr_main=2e-4, lr_backbone=1e-5),
        "C": StageHyperParams(name="C", steps=150_000, lr_main=5e-5, lr_backbone=1e-5),
    }


def _set_requires_grad(module: nn.Module, value: bool) -> None:
    for p in module.parameters():
        p.requires_grad = value


def _modules(modules: Iterable[nn.Module]) -> List[nn.Module]:
    return list(modules)


def configure_training_stage(model: ThinkJEPAVLAModel, stage: str) -> None:
    """
    Stage policy from README:
    - A: JEPA predictor + VL FiLM adapters
    - B: JEPA2AC adapter + DINO projector + FM head
    - C: Joint fine-tune
    """
    stage = stage.upper()
    if stage not in {"A", "B", "C"}:
        raise ValueError(f"Unknown stage: {stage}")

    for p in model.parameters():
        p.requires_grad = False

    if stage == "A":
        train_modules = _modules(
            [
                model.predictor_blocks,
                model.film_mlps,
                model.vl_proj,
                model.future_decoder_attn,
                model.future_out,
            ]
        )
    elif stage == "B":
        train_modules = _modules(
            [
                model.jepa_action_attn,
                model.jepa_action_queries,
                model.cond_encoder,
                model.segment_embed,
                model.special_bos,
                model.special_jsep,
                model.special_dsep,
                model.special_ssep,
                model.action_query_tokens,
                model.flow_head,
            ]
        )
        if model.use_dino:
            train_modules += _modules([model.dino_proj, model.dino_attn, model.dino_queries])
    else:
        # C: joint fine-tune everything EXCEPT JEPA backbone projections.
        # jepa_visual_proj / state_proj / action_hist_proj must stay frozen —
        # unfreezing them causes predictor collapse (flow gradients dominate
        # latent loss 50:1 and destroy pretrained representations).
        train_modules = _modules(
            [
                model.predictor_blocks,
                model.film_mlps,
                model.vl_proj,
                model.future_decoder_attn,
                model.future_out,
                model.predictor_out_norm,
                model.jepa_action_attn,
                model.jepa_action_queries,
                model.cond_encoder,
                model.cond_out_norm,
                model.segment_embed,
                model.special_bos,
                model.special_jsep,
                model.special_dsep,
                model.special_ssep,
                model.action_query_tokens,
                model.flow_head,
            ]
        )
        if model.use_dino:
            train_modules += _modules([model.dino_proj, model.dino_attn, model.dino_queries])

    for module in train_modules:
        if isinstance(module, nn.Parameter):
            module.requires_grad = True
        else:
            _set_requires_grad(module, True)


def build_optimizer(model: ThinkJEPAVLAModel, cfg: VLAConfig, stage: str) -> torch.optim.Optimizer:
    hparams = default_stage_hparams()[stage.upper()]

    main_params = []
    backbone_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "jepa_visual_proj" in name or "state_proj" in name or "action_hist_proj" in name:
            backbone_params.append(p)
        else:
            main_params.append(p)

    param_groups = []
    if main_params:
        param_groups.append({"params": main_params, "lr": hparams.lr_main, "weight_decay": hparams.weight_decay})
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": hparams.lr_backbone,
                "weight_decay": hparams.weight_decay,
            }
        )
    return torch.optim.AdamW(param_groups)
