#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

from vla_model.schema import VLAConfig
from vla_model.model import ThinkJEPAVLAModel
from vla_model.rtc import RTCConfig, RTCExecutor
from vla_model.task_splits import EVO1_HARD_TASKS, load_evo1_level_tasks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Online MetaWorld eval with RTC execution.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model-config", type=str, default="configs/train_vla_metaworld.yaml")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--task-level", type=str, default="hard")
    p.add_argument("--mt50-order-json", type=str, default="")
    p.add_argument("--tasks-json", type=str, default="")
    p.add_argument("--episodes-per-task", type=int, default=10)
    p.add_argument("--max-steps-per-episode", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--rtc-execute-steps", type=int, default=4)
    p.add_argument("--rtc-execution-horizon", type=int, default=4)
    p.add_argument("--rtc-max-guidance-weight", type=float, default=10.0)
    p.add_argument("--flow-steps", type=int, default=24)
    p.add_argument("--output-json", type=str, default="")
    return p.parse_args()


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_tasks(args: argparse.Namespace) -> List[str]:
    if args.tasks_json:
        with open(args.tasks_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [str(x) for x in data]
    if args.mt50_order_json:
        return load_evo1_level_tasks(args.mt50_order_json, level=args.task_level)
    if args.task_level.lower() == "hard":
        return list(EVO1_HARD_TASKS)
    raise ValueError("Provide --tasks-json or --mt50-order-json for non-hard levels.")


def _build_model(model_cfg_path: str, checkpoint_path: str, device: torch.device) -> tuple[ThinkJEPAVLAModel, VLAConfig]:
    cfg_raw = _load_yaml(model_cfg_path)
    mcfg = VLAConfig(**cfg_raw.get("model", cfg_raw))
    model = ThinkJEPAVLAModel(mcfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, mcfg


def _resize_image(image: np.ndarray, size: int) -> np.ndarray:
    # nearest-neighbor fallback without cv2 dependency
    h, w = image.shape[:2]
    ys = (np.linspace(0, h - 1, size)).astype(np.int64)
    xs = (np.linspace(0, w - 1, size)).astype(np.int64)
    return image[np.ix_(ys, xs)]


def _patchify(frames: torch.Tensor, out_dim: int, patch_size: int = 16) -> torch.Tensor:
    # frames: [T,H,W,3] float
    x = frames.permute(0, 3, 1, 2).contiguous()
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
    p = unfold(x).transpose(1, 2)
    proj = torch.randn(p.shape[-1], out_dim, device=p.device, dtype=p.dtype) / np.sqrt(p.shape[-1])
    return p @ proj


@dataclass
class ObsBuffer:
    images: deque
    states: deque
    actions: deque


def _make_envs():
    try:
        import metaworld  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("metaworld is not installed. Install it to run online eval.") from exc
    return metaworld


def _task_env_map(mt) -> Dict[str, object]:
    env_map = {}
    benchmark = mt.MT50(seed=0)
    for name, cls in benchmark.train_classes.items():
        env_map[name] = cls
    return env_map


def _task_obj_map(mt) -> Dict[str, object]:
    benchmark = mt.MT50(seed=0)
    out = defaultdict(list)
    for task in benchmark.train_tasks:
        out[task.env_name].append(task)
    return out


@torch.no_grad()
def run_eval(args: argparse.Namespace) -> Dict:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    tasks = _load_tasks(args)
    model, mcfg = _build_model(args.model_config, args.checkpoint, device)
    rtc = RTCExecutor(
        model,
        RTCConfig(
            chunk_horizon=mcfg.action_horizon,
            execute_steps=args.rtc_execute_steps,
            execution_horizon=args.rtc_execution_horizon,
            max_guidance_weight=args.rtc_max_guidance_weight,
            prefix_attention_schedule="exp",
        ),
    )

    mt = _make_envs()
    env_classes = _task_env_map(mt)
    task_pool = _task_obj_map(mt)

    rng = np.random.default_rng(args.seed)
    results = {"per_task": {}, "overall": {}}
    all_success = []
    all_returns = []

    for task_name in tasks:
        if task_name not in env_classes:
            continue
        env = env_classes[task_name](render_mode="rgb_array")
        task_list = task_pool[task_name]

        succ = []
        rets = []
        for epi in range(args.episodes_per_task):
            task = task_list[int(rng.integers(0, len(task_list)))]
            env.set_task(task)
            obs, _ = env.reset(seed=int(rng.integers(0, 10_000_000)))

            buf = ObsBuffer(
                images=deque(maxlen=max(mcfg.jepa_history, 32)),
                states=deque(maxlen=max(mcfg.state_history, 64)),
                actions=deque(maxlen=max(mcfg.action_history, 64)),
            )
            rtc.reset()
            step_ret = 0.0
            done = False
            success_flag = 0.0
            prev_action = np.zeros((mcfg.action_dim,), dtype=np.float32)

            for _ in range(args.max_steps_per_episode):
                frame = env.render()
                frame = _resize_image(frame, args.image_size).astype(np.float32) / 255.0
                state = np.asarray(obs, dtype=np.float32)
                buf.images.append(frame)
                buf.states.append(state)
                buf.actions.append(prev_action.copy())

                if len(buf.images) < mcfg.jepa_history or len(buf.states) < mcfg.state_history or len(buf.actions) < mcfg.action_history:
                    action = np.zeros((mcfg.action_dim,), dtype=np.float32)
                    obs, rew, term, trunc, info = env.step(action)
                    prev_action = action
                    step_ret += float(rew)
                    if bool(info.get("success", 0.0)):
                        success_flag = 1.0
                    if term or trunc:
                        done = True
                        break
                    continue

                imgs = torch.tensor(np.stack(list(buf.images), axis=0), dtype=torch.float32, device=device)
                states = torch.tensor(np.stack(list(buf.states), axis=0), dtype=torch.float32, device=device)
                actions = torch.tensor(np.stack(list(buf.actions), axis=0), dtype=torch.float32, device=device)

                frames_hist = imgs[-mcfg.jepa_history :]
                frame_cur = imgs[-1:]

                jepa_visual = _patchify(frames_hist, mcfg.jepa_in_dim)
                dino_current = _patchify(frame_cur, mcfg.dino_in_dim)[0]
                # simple sparse VL feature from latest frames
                vl_src = imgs[-32:] if imgs.shape[0] >= 32 else imgs
                idx = torch.linspace(0, vl_src.shape[0] - 1, steps=8, device=device).long()
                vl_sparse = vl_src[idx]
                vl_features = _patchify(vl_sparse, mcfg.vl_in_dim).mean(dim=1)

                batch = {
                    "jepa_visual": jepa_visual.unsqueeze(0),
                    "state_hist": states[-mcfg.state_history :].unsqueeze(0),
                    "action_hist": actions[-mcfg.action_history :].unsqueeze(0),
                    "dino_current": dino_current.unsqueeze(0),
                    "vl_features": vl_features.unsqueeze(0),
                }
                exec_actions, _ = rtc.step(batch, delay_steps=0, num_flow_steps=args.flow_steps)
                exec_np = exec_actions[0].detach().cpu().numpy()

                # Execute half chunk sequentially.
                for i in range(exec_np.shape[0]):
                    action = exec_np[i].astype(np.float32)
                    obs, rew, term, trunc, info = env.step(action)
                    prev_action = action
                    step_ret += float(rew)
                    if bool(info.get("success", 0.0)):
                        success_flag = 1.0
                    if term or trunc:
                        done = True
                        break
                if done:
                    break

            succ.append(success_flag)
            rets.append(step_ret)

        task_success = float(np.mean(succ)) if succ else float("nan")
        task_return = float(np.mean(rets)) if rets else float("nan")
        results["per_task"][task_name] = {
            "success_rate": task_success,
            "avg_return": task_return,
            "episodes": len(succ),
        }
        all_success.extend(succ)
        all_returns.extend(rets)
        env.close()

    results["overall"] = {
        "success_rate": float(np.mean(all_success)) if all_success else float("nan"),
        "avg_return": float(np.mean(all_returns)) if all_returns else float("nan"),
        "num_tasks": len(results["per_task"]),
        "episodes_total": len(all_success),
    }
    return results


def main() -> int:
    args = parse_args()
    results = run_eval(args)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
