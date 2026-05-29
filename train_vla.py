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

from vla_model.data import SequenceConfig
from vla_model.lerobot_metaworld_dataset import LeRobotDatasetConfig, LeRobotMetaWorldDataset
from vla_model.metaworld_dataset import DatasetConfig, MetaWorldVLADataset
from vla_model.model import ThinkJEPAVLAModel
from vla_model.schema import SchedulerConfig, VJEPA2Config, VLAConfig
from vla_model.task_splits import EVO1_HARD_TASKS, load_evo1_level_tasks
from vla_model.train_stages import build_optimizer, configure_training_stage


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train JEPA-driven VLA with RTC-aligned scheduling.")
    p.add_argument("--data-root", type=str, required=True, help="Dataset root.")
    p.add_argument("--config", type=str, default="configs/train_vla_metaworld.yaml")
    p.add_argument("--stage", type=str, required=True, choices=["A", "B", "C"])
    p.add_argument("--dataset-backend", type=str, default="", choices=["", "npz", "lerobot"])
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--max-train-episodes", type=int, default=0)
    p.add_argument("--max-val-episodes", type=int, default=0)
    p.add_argument("--log-every", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--eval-batches", type=int, default=0)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--outdir", type=str, default="")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--num-workers", type=int, default=-1)
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--hard-tasks-json", type=str, default="")
    p.add_argument("--task-level", type=str, default="")
    p.add_argument("--mt50-order-json", type=str, default="")
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
    model_raw = dict(raw.get("model", raw))
    scheduler_raw = model_raw.pop("scheduler", {})
    vjepa2_raw = model_raw.pop("vjepa2", {})
    model_raw["scheduler"] = SchedulerConfig(**scheduler_raw)
    model_raw["vjepa2"] = VJEPA2Config(**vjepa2_raw)
    return VLAConfig(**model_raw)


def build_seq_cfg(model_cfg: VLAConfig) -> SequenceConfig:
    sched = model_cfg.scheduler
    return SequenceConfig(
        jepa_window_frames=sched.jepa_window_frames,
        jepa_refresh_frames=sched.jepa_refresh_frames,
        vl_refresh_frames=sched.vl_refresh_frames,
        actions_per_frame=sched.actions_per_frame,
        action_horizon=model_cfg.action_horizon,
        robot_hist_frame_offsets=sched.robot_hist_frame_offsets,
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
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def compute_stage_loss(
    out,
    stage_cfg: Dict,
    require_grad: bool = False,
) -> Dict[str, Tensor]:
    loss_cfg = stage_cfg["loss"]
    optimize_action = bool(loss_cfg.get("optimize_action", True))
    optimize_jepa_pred = bool(loss_cfg.get("optimize_jepa_pred", True))
    lambda_action = float(loss_cfg.get("lambda_action", 1.0))
    lambda_jepa_pred = float(loss_cfg.get("lambda_jepa_pred", 1.0))

    device = out.predicted_jepa_targets.device
    zero = torch.zeros((), device=device)

    action_loss = zero if out.flow is None else out.flow.loss
    jepa_pred_loss = torch.nn.functional.mse_loss(out.predicted_jepa_targets, out.target_jepa_targets)

    total = zero
    if optimize_action:
        total = total + lambda_action * action_loss
    if optimize_jepa_pred:
        total = total + lambda_jepa_pred * jepa_pred_loss

    if require_grad and not total.requires_grad:
        raise RuntimeError("Total loss has no gradient path. Check stage loss flags.")

    return {
        "total": total,
        "action": action_loss.detach(),
        "jepa_pred": jepa_pred_loss.detach(),
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
    acc_action = 0.0
    acc_jepa = 0.0
    n = 0
    for batch in val_loader:
        if n >= eval_batches:
            break
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(batch, sample_actions=False)
            losses = compute_stage_loss(out, stage_cfg, require_grad=False)
        acc_total += float(losses["total"])
        acc_action += float(losses["action"])
        acc_jepa += float(losses["jepa_pred"])
        n += 1
    model.train()
    if n == 0:
        return {"total": float("nan"), "action": float("nan"), "jepa_pred": float("nan")}
    return {"total": acc_total / n, "action": acc_action / n, "jepa_pred": acc_jepa / n}


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
    keep: int = 2,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "stage_cfg": stage_cfg,
            "model_cfg": model_cfg,
        },
        outdir / f"ckpt_step_{step:07d}.pt",
    )
    ckpts = sorted(outdir.glob("ckpt_step_*.pt"), key=lambda p: p.stat().st_mtime)
    for old in ckpts[:-keep]:
        old.unlink()


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
        train_ds = MetaWorldVLADataset(
            DatasetConfig(
                root_dir=args.data_root,
                split="train",
                val_ratio=float(dataset_cfg_root.get("split", {}).get("val_ratio", 0.1)),
                seed=seed,
                hard_tasks=task_filter,
                max_episodes=args.max_train_episodes if args.max_train_episodes > 0 else None,
            ),
            model_cfg,
            seq_cfg,
        )
        val_ds = MetaWorldVLADataset(
            DatasetConfig(
                root_dir=args.data_root,
                split="val",
                val_ratio=float(dataset_cfg_root.get("split", {}).get("val_ratio", 0.1)),
                seed=seed,
                hard_tasks=task_filter,
                max_episodes=args.max_val_episodes if args.max_val_episodes > 0 else None,
            ),
            model_cfg,
            seq_cfg,
        )
    else:
        lcfg = dataset_cfg_root.get("lerobot", {})
        train_ds = LeRobotMetaWorldDataset(
            LeRobotDatasetConfig(
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
                episode_cache_size=int(lcfg.get("episode_cache_size", 8)),
            ),
            model_cfg,
            seq_cfg,
        )
        val_ds = LeRobotMetaWorldDataset(
            LeRobotDatasetConfig(
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
                episode_cache_size=int(lcfg.get("episode_cache_size", 8)),
            ),
            model_cfg,
            seq_cfg,
        )

    device_arg = args.device if args.device else str(runtime_cfg.get("device", "cuda"))
    num_workers = args.num_workers if args.num_workers >= 0 else int(runtime_cfg.get("num_workers", 4))
    pin_memory = (device_arg != "cpu") and (num_workers > 0)
    global_batch_size = int(stage_yaml.get("global_batch_size", 16))
    train_loader = DataLoader(
        train_ds,
        batch_size=global_batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=global_batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=0 if num_workers == 0 else max(1, num_workers // 2),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    device = torch.device(device_arg if torch.cuda.is_available() or device_arg == "cpu" else "cpu")
    model = ThinkJEPAVLAModel(model_cfg).to(device)
    stage_name = str(args.stage).upper()
    configure_training_stage(model, stage_name)
    optimizer = build_optimizer(model, model_cfg, stage_name)

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except ValueError:
            print("[resume] Optimizer state mismatch, starting fresh optimizer")

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

    model.train()
    t0 = time.time()
    train_iter = cycle(train_loader)

    for step in range(1, max_steps + 1):
        batch = move_batch_to_device(next(train_iter), device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(batch, sample_actions=False)
            losses = compute_stage_loss(out, {"loss": stage_yaml["loss"]}, require_grad=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(runtime_cfg.get("grad_clip_norm", 1.0)))
        optimizer.step()

        if step % log_every == 0:
            dt = time.time() - t0
            sps = step / max(dt, 1e-6)
            print(
                f"[train] step={step}/{max_steps} total={float(losses['total'].detach()):.6f} "
                f"action={float(losses['action']):.6f} jepa_pred={float(losses['jepa_pred']):.6f} sps={sps:.2f}",
                flush=True,
            )

        if step % eval_every == 0:
            metrics = evaluate(model, val_loader, {"loss": stage_yaml["loss"]}, device, eval_batches, amp_dtype)
            print(
                f"[eval] step={step} total={metrics['total']:.6f} action={metrics['action']:.6f} "
                f"jepa_pred={metrics['jepa_pred']:.6f}",
                flush=True,
            )

        if step % save_every == 0:
            save_checkpoint(outdir, step, model, optimizer, stage_yaml, model_cfg)

    save_checkpoint(outdir, max_steps, model, optimizer, stage_yaml, model_cfg)
    print(f"[done] stage={stage_name} steps={max_steps} outdir={outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
