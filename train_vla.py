#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader

from vla_model.schema import VLAConfig
from vla_model.data import SequenceConfig
from vla_model.lerobot_metaworld_dataset import LeRobotDatasetConfig, LeRobotMetaWorldDataset
from vla_model.metaworld_dataset import DatasetConfig, MetaWorldVLADataset
from vla_model.model import ThinkJEPAVLAModel
from vla_model.task_splits import EVO1_HARD_TASKS, load_evo1_level_tasks
from vla_model.train_stages import configure_training_stage


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VL-guided JEPA VLA with FM head.")
    p.add_argument("--data-root", type=str, required=True, help="Dataset root.")
    p.add_argument("--config", type=str, default="configs/train_vla_metaworld.yaml")
    p.add_argument("--stage", type=str, required=True, choices=["A", "B", "C"])
    p.add_argument("--dataset-backend", type=str, default="", choices=["", "npz", "lerobot"])
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--max-train-episodes", type=int, default=0)
    p.add_argument("--max-val-episodes", type=int, default=0)
    p.add_argument("--log-every", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0, help="Override stage config train.steps when > 0.")
    p.add_argument("--eval-every", type=int, default=0, help="Override stage config eval_every_steps when > 0.")
    p.add_argument("--eval-batches", type=int, default=0)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--outdir", type=str, default="")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--num-workers", type=int, default=-1)
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--hard-tasks-json", type=str, default="")
    p.add_argument("--task-level", type=str, default="", help="Use split level from mt50 order, e.g. hard.")
    p.add_argument("--mt50-order-json", type=str, default="", help="Path to mt50_order.json.")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_cfg(raw: Dict) -> VLAConfig:
    model_raw = raw.get("model", raw)
    return VLAConfig(**model_raw)


def build_seq_cfg(model_cfg: VLAConfig) -> SequenceConfig:
    return SequenceConfig(
        frame_history=model_cfg.jepa_history,
        state_history=model_cfg.state_history,
        action_history=model_cfg.action_history,
        action_horizon=model_cfg.action_horizon,
    )


def parse_hard_tasks(path: str) -> Optional[list[str]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("hard-tasks-json must be a JSON list of task names.")
    return [str(x) for x in data]


def resolve_task_filter(args: argparse.Namespace) -> Optional[list[str]]:
    if args.hard_tasks_json:
        return parse_hard_tasks(args.hard_tasks_json)
    if args.task_level:
        level = args.task_level.lower()
        if level in {"all", "none", "off"}:
            return None
        if args.mt50_order_json:
            return load_evo1_level_tasks(args.mt50_order_json, level=level)
        if level == "hard":
            return list(EVO1_HARD_TASKS)
    return None


def move_batch_to_device(batch: Dict[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            if v.device != device:
                out[k] = v.to(device, non_blocking=True)
            else:
                out[k] = v
        else:
            out[k] = v
    return out


def make_optimizer(model: ThinkJEPAVLAModel, stage_cfg: Dict) -> torch.optim.Optimizer:
    optim_raw = stage_cfg["optimizer"]
    lr_main = float(optim_raw["lr_main"])
    lr_backbone = float(optim_raw["lr_backbone"])
    weight_decay = float(optim_raw.get("weight_decay", 0.05))

    main_params = []
    backbone_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "jepa_visual_proj" in name or "state_proj" in name or "action_hist_proj" in name:
            backbone_params.append(p)
        else:
            main_params.append(p)

    groups = []
    if main_params:
        groups.append({"params": main_params, "lr": lr_main, "weight_decay": weight_decay})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay})
    return torch.optim.AdamW(groups)


def compute_stage_loss(
    out,
    batch: Dict[str, Tensor],
    stage_cfg: Dict,
    require_grad: bool = False,
) -> Dict[str, Tensor]:
    loss_cfg = stage_cfg["loss"]
    optimize_flow = bool(loss_cfg.get("optimize_flow", True))
    optimize_jepa_latent = bool(loss_cfg.get("optimize_jepa_latent", True))
    lambda_jepa = float(loss_cfg.get("lambda_jepa_latent", 0.05))

    device = batch["jepa_visual"].device
    zero = torch.zeros((), device=device)

    flow_loss = zero
    if out.flow is not None:
        flow_loss = out.flow.loss

    latent_loss = zero
    if "target_future_latent" in batch:
        latent_loss = torch.nn.functional.mse_loss(out.pred_future_latent, batch["target_future_latent"])

    total = zero
    if optimize_flow:
        total = total + flow_loss
    if optimize_jepa_latent:
        total = total + lambda_jepa * latent_loss

    if require_grad and not total.requires_grad:
        raise RuntimeError(
            "Total loss has no gradient path. Check stage loss flags and target availability."
        )

    return {
        "total": total,
        "flow": flow_loss.detach(),
        "latent": latent_loss.detach(),
    }


@torch.no_grad()
def evaluate(
    model: ThinkJEPAVLAModel,
    val_loader: DataLoader,
    stage_cfg: Dict,
    device: torch.device,
    eval_batches: int,
    amp_dtype: Optional[torch.dtype],
) -> Dict[str, float]:
    model.eval()
    acc_total = 0.0
    acc_flow = 0.0
    acc_latent = 0.0
    n = 0
    iterator = iter(val_loader)
    while n < eval_batches:
        try:
            batch = next(iterator)
        except StopIteration:
            break
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(batch, sample_actions=False)
            losses = compute_stage_loss(out, batch, stage_cfg, require_grad=False)
        acc_total += float(losses["total"])
        acc_flow += float(losses["flow"])
        acc_latent += float(losses["latent"])
        n += 1
    model.train()
    if n == 0:
        return {"total": float("nan"), "flow": float("nan"), "latent": float("nan")}
    return {"total": acc_total / n, "flow": acc_flow / n, "latent": acc_latent / n}


def cycle(loader: DataLoader) -> Iterable:
    while True:
        for x in loader:
            yield x


def save_checkpoint(
    outdir: Path,
    step: int,
    model: ThinkJEPAVLAModel,
    optimizer: torch.optim.Optimizer,
    stage_cfg: Dict,
    model_cfg: VLAConfig,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "stage_cfg": stage_cfg,
        "model_cfg": model_cfg.__dict__,
    }
    torch.save(ckpt, outdir / f"ckpt_step_{step:07d}.pt")


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    runtime_cfg = cfg.get("runtime", {})
    dataset_cfg_root = cfg.get("dataset", {})
    task_cfg = cfg.get("task_filter", {})
    all_stages = cfg.get("stages", {})
    if args.stage not in all_stages:
        raise KeyError(f"Stage `{args.stage}` not in config stages: {list(all_stages.keys())}")
    stage_yaml = all_stages[args.stage]

    seed = args.seed if args.seed >= 0 else int(runtime_cfg.get("seed", 42))
    set_seed(seed)

    model_cfg = build_model_cfg(cfg)
    seq_cfg = build_seq_cfg(model_cfg)

    backend = args.dataset_backend.lower() if args.dataset_backend else str(dataset_cfg_root.get("backend", "npz")).lower()
    task_filter = resolve_task_filter(args)
    cli_task_override = bool(args.hard_tasks_json or args.task_level or args.mt50_order_json)
    if task_filter is None and not cli_task_override:
        cfg_level = str(task_cfg.get("level", "")).strip().lower()
        cfg_hard_json = str(task_cfg.get("hard_tasks_json", "")).strip()
        cfg_order = str(task_cfg.get("mt50_order_json", "")).strip()
        if cfg_hard_json:
            with open(cfg_hard_json, "r", encoding="utf-8") as f:
                task_filter = [str(x) for x in json.load(f)]
        elif cfg_level:
            if cfg_order:
                task_filter = load_evo1_level_tasks(cfg_order, level=cfg_level)
            elif cfg_level == "hard":
                task_filter = list(EVO1_HARD_TASKS)

    if backend not in {"npz", "lerobot"}:
        raise ValueError(f"Unsupported dataset backend: {backend}")

    if backend == "npz":
        npz_cfg = dataset_cfg_root.get("npz", {})
        train_ds_cfg = DatasetConfig(
            root_dir=args.data_root,
            split="train",
            val_ratio=float(dataset_cfg_root.get("split", {}).get("val_ratio", 0.1)),
            seed=seed,
            hard_tasks=task_filter,
            max_episodes=args.max_train_episodes if args.max_train_episodes > 0 else None,
            require_precomputed_features=bool(npz_cfg.get("require_precomputed_features", False)),
        )
        val_ds_cfg = DatasetConfig(
            root_dir=args.data_root,
            split="val",
            val_ratio=float(dataset_cfg_root.get("split", {}).get("val_ratio", 0.1)),
            seed=seed,
            hard_tasks=task_filter,
            max_episodes=args.max_val_episodes if args.max_val_episodes > 0 else None,
            require_precomputed_features=bool(npz_cfg.get("require_precomputed_features", False)),
        )
        train_ds = MetaWorldVLADataset(train_ds_cfg, model_cfg, seq_cfg)
        val_ds = MetaWorldVLADataset(val_ds_cfg, model_cfg, seq_cfg)
    else:
        lcfg = dataset_cfg_root.get("lerobot", {})
        train_ds_cfg = LeRobotDatasetConfig(
            root_dir=args.data_root,
            split="train",
            val_ratio=float(dataset_cfg_root.get("split", {}).get("val_ratio", 0.1)),
            seed=seed,
            hard_tasks=task_filter,
            max_episodes=args.max_train_episodes if args.max_train_episodes > 0 else None,
            frame_key=str(lcfg.get("frame_key", "observation.images.image")),
            states_key=str(lcfg.get("states_key", "observation.state")),
            action_key=str(lcfg.get("action_key", "action")),
            use_stats_norm=bool(lcfg.get("use_stats_norm", True)),
            decode_video=bool(lcfg.get("decode_video", True)),
            episode_cache_size=int(lcfg.get("episode_cache_size", 64)),
        )
        val_ds_cfg = LeRobotDatasetConfig(
            root_dir=args.data_root,
            split="val",
            val_ratio=float(dataset_cfg_root.get("split", {}).get("val_ratio", 0.1)),
            seed=seed,
            hard_tasks=task_filter,
            max_episodes=args.max_val_episodes if args.max_val_episodes > 0 else None,
            frame_key=str(lcfg.get("frame_key", "observation.images.image")),
            states_key=str(lcfg.get("states_key", "observation.state")),
            action_key=str(lcfg.get("action_key", "action")),
            use_stats_norm=bool(lcfg.get("use_stats_norm", True)),
            decode_video=bool(lcfg.get("decode_video", True)),
            episode_cache_size=int(lcfg.get("episode_cache_size", 64)),
        )
        train_ds = LeRobotMetaWorldDataset(train_ds_cfg, model_cfg, seq_cfg)
        val_ds = LeRobotMetaWorldDataset(val_ds_cfg, model_cfg, seq_cfg)
    device_arg = args.device if args.device else str(runtime_cfg.get("device", "cuda"))
    num_workers = args.num_workers if args.num_workers >= 0 else int(runtime_cfg.get("num_workers", 4))
    pin_memory = (device_arg != "cpu") and (num_workers > 0)
    global_batch_size = int(stage_yaml.get("global_batch_size", 128))

    prefetch_factor = 4
    persistent_workers = num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=global_batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=global_batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=0 if num_workers == 0 else max(1, num_workers // 2),
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    device = torch.device(device_arg if torch.cuda.is_available() or device_arg == "cpu" else "cpu")
    model = ThinkJEPAVLAModel(model_cfg).to(device)
    stage_name = str(args.stage).upper()
    configure_training_stage(model, stage_name)
    optimizer = make_optimizer(model, {"optimizer": stage_yaml["optimizer"]})

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except ValueError:
            print(f"[resume] Optimizer state mismatch (cross-stage), starting fresh optimizer")

    precision = str(runtime_cfg.get("precision", "bf16")).lower()
    if device.type == "cuda" and precision == "bf16":
        amp_dtype = torch.bfloat16
    elif device.type == "cuda" and precision == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = None

    max_steps = int(args.max_steps) if args.max_steps > 0 else int(stage_yaml["steps"])
    eval_every = int(args.eval_every) if args.eval_every > 0 else int(stage_yaml["eval_every_steps"])
    log_every = int(args.log_every) if args.log_every > 0 else int(runtime_cfg.get("log_every", 100))
    save_every = int(args.save_every) if args.save_every > 0 else int(runtime_cfg.get("save_every", 5000))
    eval_batches = int(args.eval_batches) if args.eval_batches > 0 else int(runtime_cfg.get("eval_batches", 50))

    outdir_base = args.outdir if args.outdir else str(runtime_cfg.get("outdir", "outputs"))
    outdir = Path(outdir_base) / f"stage_{stage_name.lower()}"
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "effective_stage.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"stage": stage_name, **stage_yaml}, f, sort_keys=False)
    with open(outdir / "effective_model.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"model": model_cfg.__dict__}, f, sort_keys=False)

    model.train()
    t0 = time.time()
    train_iter = cycle(train_loader)

    for step in range(1, max_steps + 1):
        batch = next(train_iter)
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(batch, sample_actions=False)
            losses = compute_stage_loss(out, batch, {"loss": stage_yaml["loss"]}, require_grad=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(runtime_cfg.get("grad_clip_norm", 1.0)))
        optimizer.step()

        if step % log_every == 0:
            dt = time.time() - t0
            steps_per_sec = step / max(dt, 1e-6)
            print(
                f"[train] step={step}/{max_steps} total={float(losses['total'].detach()):.6f} "
                f"flow={float(losses['flow']):.6f} latent={float(losses['latent']):.6f} sps={steps_per_sec:.2f}",
                flush=True,
            )

        if step % eval_every == 0:
            metrics = evaluate(model, val_loader, {"loss": stage_yaml["loss"]}, device, eval_batches, amp_dtype)
            print(
                f"[eval] step={step} total={metrics['total']:.6f} flow={metrics['flow']:.6f} latent={metrics['latent']:.6f}",
                flush=True,
            )

        if step % save_every == 0:
            save_checkpoint(outdir, step, model, optimizer, stage_yaml, model_cfg)

    save_checkpoint(outdir, max_steps, model, optimizer, stage_yaml, model_cfg)
    print(f"[done] stage={stage_name} steps={max_steps} outdir={outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
