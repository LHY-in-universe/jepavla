#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import yaml
from PIL import Image

from vla_model.model import ThinkJEPAVLAModel
from vla_model.rtc import RTCConfig, RTCExecutor
from vla_model.scheduler import ControlScheduler
from vla_model.schema import SchedulerConfig, VJEPA2Config, VLAConfig
from vla_model.task_splits import EVO1_HARD_TASKS, load_evo1_level_tasks


FEATURE_LAYERS = [7, 14, 21, 27]


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
    p.add_argument("--flow-steps", type=int, default=24)
    p.add_argument("--output-json", type=str, default="")
    return p.parse_args()


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_model(model_cfg_path: str, checkpoint_path: str, device: torch.device) -> tuple[ThinkJEPAVLAModel, VLAConfig]:
    cfg_raw = _load_yaml(model_cfg_path)
    model_raw = dict(cfg_raw.get("model", cfg_raw))
    model_raw["scheduler"] = SchedulerConfig(**model_raw.pop("scheduler", {}))
    model_raw["vjepa2"] = VJEPA2Config(**model_raw.pop("vjepa2", {}))
    mcfg = VLAConfig(**model_raw)
    model = ThinkJEPAVLAModel(mcfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model, mcfg


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


def _load_qwen_vl(device: torch.device):
    from pathlib import Path
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    cache_dir = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots"
    snapshots = sorted(cache_dir.glob("*"))
    if not snapshots:
        raise FileNotFoundError(f"No Qwen3-VL snapshot found in {cache_dir}")
    local_path = str(snapshots[-1])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        local_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(local_path)
    model.eval()
    return model, processor


@torch.no_grad()
def _extract_vl_features(qwen_model, processor, frame: np.ndarray, device: torch.device) -> torch.Tensor:
    pil_img = Image.fromarray(frame)
    messages = [{"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": "x"}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    inputs = processor(text=[text], images=[pil_img], return_tensors="pt")
    for k, v in inputs.items():
        if v.dtype in (torch.int64, torch.long):
            inputs[k] = v.to(device)
        else:
            inputs[k] = v.to(device, dtype=torch.bfloat16)
    outputs = qwen_model(**inputs, output_hidden_states=True)
    img_token_id = qwen_model.config.image_token_id
    img_mask = inputs["input_ids"][0] == img_token_id
    layer_feats = []
    for layer_idx in FEATURE_LAYERS:
        h = outputs.hidden_states[layer_idx]
        img_feats = h[0, img_mask, :]
        layer_feats.append(img_feats.mean(dim=0).float())
    return torch.stack(layer_feats)[None]


@dataclass
class ObsBuffer:
    images: deque
    sparse_images: deque
    sparse_states: deque
    sparse_actions: deque


def _make_envs():
    try:
        import metaworld  # type: ignore
    except Exception as exc:
        raise RuntimeError("metaworld is not installed. Install it to run online eval.") from exc
    return metaworld


def _task_env_map(mt) -> Dict[str, object]:
    benchmark = mt.MT50(seed=0)
    return {name: cls for name, cls in benchmark.train_classes.items()}


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
    scheduler = ControlScheduler(mcfg.scheduler)
    rtc = RTCExecutor(
        model,
        RTCConfig(
            chunk_horizon=mcfg.action_horizon,
            execute_steps=mcfg.rtc_execute_steps,
            execution_horizon=mcfg.rtc_execution_horizon,
            max_guidance_weight=mcfg.rtc_max_guidance_weight,
            prefix_attention_schedule="exp",
        ),
    )
    qwen_model, qwen_processor = _load_qwen_vl(device)
    mt = _make_envs()
    env_classes = _task_env_map(mt)
    task_pool = _task_obj_map(mt)

    rng = np.random.default_rng(args.seed)
    results = {"per_task": {}, "overall": {}}
    all_success = []
    all_returns = []
    t_start = time.time()

    hist_offsets = list(mcfg.scheduler.robot_hist_frame_offsets)
    max_hist = max(hist_offsets)
    for task_idx, task_name in enumerate(tasks):
        if task_name not in env_classes:
            continue
        env = env_classes[task_name](render_mode="rgb_array")
        task_list = task_pool[task_name]
        succ = []
        rets = []
        for _ in range(args.episodes_per_task):
            task = task_list[int(rng.integers(0, len(task_list)))]
            env.set_task(task)
            obs, _ = env.reset(seed=int(rng.integers(0, 10_000_000)))
            rtc.reset()
            action_queue: list[np.ndarray] = []
            prev_action = np.zeros((mcfg.action_dim,), dtype=np.float32)
            step_ret = 0.0
            success_flag = 0.0
            vl_cond = torch.zeros(1, 1, mcfg.vl_in_dim, device=device)
            buf = ObsBuffer(
                images=deque(maxlen=mcfg.scheduler.jepa_window_frames),
                sparse_images=deque(maxlen=max_hist + 1),
                sparse_states=deque(maxlen=max_hist + 1),
                sparse_actions=deque(maxlen=max_hist + 1),
            )

            for frame_idx in range(args.max_steps_per_episode):
                frame = env.render()
                frame_t = torch.from_numpy(frame.astype(np.float32) / 255.0)
                state = np.asarray([obs[0], obs[1], obs[2], obs[7]], dtype=np.float32)
                buf.images.append(frame_t)
                buf.sparse_images.append(frame_t)
                buf.sparse_states.append(torch.from_numpy(state))
                buf.sparse_actions.append(torch.from_numpy(prev_action.copy()))

                if scheduler.is_vl_tick(frame_idx):
                    vl_cond = _extract_vl_features(qwen_model, qwen_processor, frame, device)

                if scheduler.is_jepa_tick(frame_idx) and len(buf.images) == mcfg.scheduler.jepa_window_frames and len(buf.sparse_images) > max_hist:
                    sparse_hist_images = []
                    sparse_hist_states = []
                    sparse_hist_actions = []
                    sparse_list_images = list(buf.sparse_images)
                    sparse_list_states = list(buf.sparse_states)
                    sparse_list_actions = list(buf.sparse_actions)
                    for offset in hist_offsets:
                        idx = max(0, len(sparse_list_images) - 1 - offset)
                        sparse_hist_images.append(sparse_list_images[idx])
                        sparse_hist_states.append(sparse_list_states[idx])
                        sparse_hist_actions.append(sparse_list_actions[idx])
                    batch = {
                        "context_frames": torch.stack(list(buf.images), dim=0)[None].to(device),
                        "sparse_hist_images": torch.stack(sparse_hist_images, dim=0)[None].to(device),
                        "sparse_hist_states": torch.stack(sparse_hist_states, dim=0)[None].to(device),
                        "sparse_hist_actions": torch.stack(sparse_hist_actions, dim=0)[None].to(device),
                        "vl_cond": vl_cond.to(device),
                    }
                    execute, _ = rtc.step(batch, num_flow_steps=args.flow_steps)
                    action_queue = [a.detach().cpu().numpy() for a in execute[0]]

                frame_actions = []
                for _ in range(mcfg.scheduler.actions_per_frame):
                    if action_queue:
                        frame_actions.append(action_queue.pop(0))
                    else:
                        frame_actions.append(np.zeros((mcfg.action_dim,), dtype=np.float32))

                action = frame_actions[-1]
                obs, rew, term, trunc, info = env.step(action)
                prev_action = action
                step_ret += float(rew)
                if bool(info.get("success", 0.0)):
                    success_flag = 1.0
                if term or trunc:
                    break

            succ.append(success_flag)
            rets.append(step_ret)
            all_success.append(success_flag)
            all_returns.append(step_ret)
        env.close()
        results["per_task"][task_name] = {"success": float(np.mean(succ)), "return": float(np.mean(rets))}

    results["overall"] = {
        "success": float(np.mean(all_success)) if all_success else 0.0,
        "return": float(np.mean(all_returns)) if all_returns else 0.0,
        "elapsed_sec": time.time() - t_start,
    }
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    return results


def main() -> int:
    args = parse_args()
    results = run_eval(args)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
