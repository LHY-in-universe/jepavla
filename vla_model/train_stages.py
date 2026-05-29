from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
from torch import nn

from .model import ThinkJEPAVLAModel
from .schema import VLAConfig


@dataclass
class StageHyperParams:
    name: str
    steps: int
    lr_main: float
    weight_decay: float = 0.05


def default_stage_hparams() -> Dict[str, StageHyperParams]:
    return {
        "A": StageHyperParams(name="A", steps=250_000, lr_main=2e-4),
        "B": StageHyperParams(name="B", steps=200_000, lr_main=1e-4),
        "C": StageHyperParams(name="C", steps=150_000, lr_main=5e-5),
    }


def _set_requires_grad(module: nn.Module, value: bool) -> None:
    for p in module.parameters():
        p.requires_grad = value


def _modules(modules: Iterable[nn.Module]) -> List[nn.Module]:
    return list(modules)


def configure_training_stage(model: ThinkJEPAVLAModel, stage: str) -> None:
    stage = stage.upper()
    if stage not in {"A", "B", "C"}:
        raise ValueError(f"Unknown stage: {stage}")

    for p in model.parameters():
        p.requires_grad = False

    if stage == "A":
        train_modules = _modules([model.sparse_hist_encoder, model.predictor_adapter])
    elif stage == "B":
        train_modules = _modules([model.jepa_aggregator, model.state_summary, model.flow_head])
    else:
        train_modules = _modules(
            [model.sparse_hist_encoder, model.predictor_adapter, model.jepa_aggregator, model.state_summary, model.flow_head]
        )

    for module in train_modules:
        _set_requires_grad(module, True)


def build_optimizer(model: ThinkJEPAVLAModel, cfg: VLAConfig, stage: str) -> torch.optim.Optimizer:
    hparams = default_stage_hparams()[stage.upper()]
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=hparams.lr_main, weight_decay=hparams.weight_decay)
