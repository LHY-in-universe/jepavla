#!/usr/bin/env python3
"""Pre-extract Qwen3-VL-2B LLM multi-layer features for every 9th frame of each episode.

For each sampled frame: full Qwen3-VL forward pass -> extract 4 LLM layer hidden states
at image token positions -> mean pool -> [4, 2048].

Saves one .npy per episode: [num_sampled_frames, 4, 2048] float16.

Usage:
  HF_ENDPOINT=https://hf-mirror.com python scripts/extract_qwen_vl_features.py \
    --dataset-root /path/to/Evo1_MetaWorld_Dataset \
    --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import av
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# LLM layers to extract features from
FEATURE_LAYERS = [7, 14, 21, 27]  # spread across 28-layer LLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--vl-stride", type=int, default=9)
    p.add_argument("--max-episodes", type=int, default=0)
    return p.parse_args()


def scan_episodes(dataset_root: Path) -> List[dict]:
    meta_dir = dataset_root / "meta"
    videos_dir = dataset_root / "videos"
    ep_path = meta_dir / "episodes.jsonl"
    if not ep_path.exists():
        raise FileNotFoundError(f"No episodes.jsonl in {meta_dir}")

    episodes = []
    with open(ep_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ep_idx = obj.get("episode_index", -1)
            chunk_num = ep_idx // 1000 if ep_idx >= 0 else 0
            chunk = f"chunk-{chunk_num:03d}"
            video_path = videos_dir / chunk / "observation.images.image" / f"episode_{ep_idx:06d}.mp4"
            if not video_path.exists():
                continue
            episodes.append({"episode_index": ep_idx, "video_path": str(video_path)})
    return episodes


def decode_all_frames(video_path: Path) -> List[np.ndarray]:
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    container.close()
    return frames


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_root / "vl_features_qwen3"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[extract] device={device}, stride={args.vl_stride}, layers={FEATURE_LAYERS}")

    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    print("[extract] Loading Qwen3-VL-2B-Instruct...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct", dtype=torch.bfloat16, device_map=device)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    model.eval()
    img_token_id = model.config.image_token_id

    episodes = scan_episodes(dataset_root)
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    print(f"[extract] {len(episodes)} episodes")

    success, skipped = 0, 0
    for ep in tqdm(episodes, desc="Extracting"):
        video_path = Path(ep["video_path"])
        out_path = output_dir / f"episode_{ep['episode_index']:06d}.npy"
        if out_path.exists():
            success += 1
            continue
        if not video_path.exists():
            skipped += 1
            continue

        try:
            all_frames = decode_all_frames(video_path)
            if not all_frames:
                skipped += 1
                continue

            # Sample every vl_stride-th frame
            sampled_indices = list(range(0, len(all_frames), args.vl_stride))
            sampled_frames = [all_frames[i] for i in sampled_indices]

            all_vl_features = []
            for frame_arr in sampled_frames:
                pil_img = Image.fromarray(frame_arr)
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": "x"},
                ]}]
                text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
                inputs = processor(text=[text], images=[pil_img], return_tensors="pt")

                # Move to device with correct dtypes
                for k, v in inputs.items():
                    if v.dtype in (torch.int64, torch.long):
                        inputs[k] = v.to(device)
                    else:
                        inputs[k] = v.to(device, dtype=torch.bfloat16)

                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)

                # Extract 4 layer features at image token positions
                img_mask = (inputs["input_ids"][0] == img_token_id)
                layer_feats = []
                for layer_idx in FEATURE_LAYERS:
                    h = outputs.hidden_states[layer_idx]          # [1, T, 2048]
                    img_feats = h[0, img_mask, :]                  # [225, 2048]
                    pooled = img_feats.mean(dim=0)                 # [2048]
                    layer_feats.append(pooled.float().cpu())
                all_vl_features.append(torch.stack(layer_feats))   # [4, 2048]

            features = torch.stack(all_vl_features).numpy().astype(np.float16)  # [N, 4, 2048]
        except Exception as e:
            print(f"\n  ERROR ep {ep['episode_index']}: {e}")
            skipped += 1
            continue

        np.save(out_path, features)
        success += 1

    manifest = {
        "model": "Qwen3-VL-2B-Instruct",
        "feature_type": "llm_4layer_hidden_states",
        "layers": FEATURE_LAYERS,
        "dim": 2048,
        "dtype": "float16",
        "vl_stride": args.vl_stride,
        "num_episodes": success,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    print(f"[extract] Done: {success} extracted, {skipped} skipped -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
