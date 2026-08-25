#!/usr/bin/env python
"""Run Mage-VL image or video inference offline on Apple Silicon with MLX.

Mirrors the CLI of microsoft/Mage's mage_vl/inference_base.py for the offline
mode. Online mode is intentionally absent: it exists to talk to a CUDA SGLang
server, which is the opposite of what this port is for.

The codec backend needs cv-preinfer, which has no macOS build. Point
CV_PREINFER_BIN at docker/cv-preinfer to run it through a container.
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mage_vl_mlx.model import MageVL  # noqa: E402
from mage_vl_mlx.prompt import PromptBuilder  # noqa: E402
from mage_vl_mlx.video import (  # noqa: E402
    build_patch_positions, patchify, smart_resize,
)

EOS_TOKEN_ID = 151645  # <|im_end|>


def preprocess_image(path: str, patch_size: int = 16, merge_size: int = 2,
                     min_pixels: int = 3136, max_pixels: int = 4000000):
    """Qwen2VL image preprocessing: smart_resize, then patchify."""
    image = Image.open(path).convert("RGB")
    height, width = smart_resize(
        image.height, image.width, patch_size,
        min_pixels=min_pixels, max_pixels=max_pixels,
        align_patch_size=patch_size * merge_size,
    )
    if (height, width) != (image.height, image.width):
        image = image.resize((width, height), Image.BICUBIC)
    pixel_values, grid_h, grid_w = patchify(
        [np.asarray(image)], patch_size, merge_size)
    grid = np.array([[1, grid_h, grid_w]], dtype=np.int64)
    return pixel_values, grid


def build_inputs(args, builder: PromptBuilder):
    if args.image:
        pixel_values, grid = preprocess_image(args.image)
        positions = build_patch_positions([0], int(grid[0][1]), int(grid[0][2]))
        return pixel_values, grid, positions, builder.for_image(args.question, grid)

    if args.video_backend == "codec":
        from mage_vl_mlx.codec import preprocess_codec, run_cv_preinfer

        asset_dir = run_cv_preinfer(args.video)
        processed = preprocess_codec(asset_dir)
        ids = builder.for_video_codec(
            args.question, processed["patch_positions"], processed["fps"])
    else:
        from mage_vl_mlx.video import preprocess_video

        processed = preprocess_video(
            args.video, max_frames=args.num_frames, target_fps=args.target_fps)
        ids = builder.for_video_frames(
            args.question, processed["grid_thw"], processed["frame_timestamps"])
    return (processed["pixel_values"], processed["grid_thw"],
            processed["patch_positions"], ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("offline",), default="offline")
    media = parser.add_mutually_exclusive_group(required=True)
    media.add_argument("--image", help="Local image path")
    media.add_argument("--video", help="Local video path")
    parser.add_argument("--video-backend", choices=("frames", "codec"), default="frames")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--target-fps", type=float, default=None)
    parser.add_argument("--question", default="Describe this media.")
    parser.add_argument("--weights", type=Path, default=Path("weights/mage-vl-bf16"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    dtype = getattr(mx, args.dtype)
    start = time.perf_counter()
    model = MageVL.from_pretrained(args.weights)
    if dtype != mx.bfloat16:
        model.update(model.apply(lambda p: p.astype(dtype)))
        mx.eval(model.parameters())
    load_s = time.perf_counter() - start

    builder = PromptBuilder(args.weights)
    pixel_values, grid, positions, ids = build_inputs(args, builder)

    start = time.perf_counter()
    tokens = model.generate(
        mx.array(ids),
        mx.array(pixel_values).astype(dtype),
        mx.array(np.asarray(grid).astype(np.int32)),
        mx.array(np.asarray(positions).astype(np.int32)),
        max_new_tokens=args.max_new_tokens,
        eos_token_id=EOS_TOKEN_ID,
    )
    wall_s = time.perf_counter() - start

    print(builder.decode(tokens).strip())
    if args.verbose:
        print(f"\n[load {load_s:.2f}s | prompt {ids.shape[1]} tokens | "
              f"generated {len(tokens)} in {wall_s:.2f}s "
              f"({len(tokens) / wall_s:.1f} tok/s) | "
              f"peak {mx.get_peak_memory() / 1024**3:.2f} GB]", file=sys.stderr)


if __name__ == "__main__":
    main()
