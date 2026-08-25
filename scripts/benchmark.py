"""Measure this port's image inference speed on Apple Silicon."""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mage_vl_mlx.model import MageVL  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("weights/mage-vl-bf16"))
    parser.add_argument("--fixture", type=Path,
                        default=Path("fixtures/stage1/objects_1536x1024_358kb/mps-bfloat16.npz"))
    parser.add_argument("--video", type=Path,
                        help="preprocess this video instead of using the fixture's pixels")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    t0 = time.perf_counter()
    model = MageVL.from_pretrained(args.weights)
    load_s = time.perf_counter() - t0

    fixture = np.load(args.fixture)
    input_ids = mx.array(fixture["input_ids"].astype(np.int32))
    if args.video is not None:
        from mage_vl_mlx.video import preprocess_video

        t1 = time.perf_counter()
        processed = preprocess_video(str(args.video))
        preprocess_s = time.perf_counter() - t1
        pixel_values = mx.array(processed["pixel_values"]).astype(mx.bfloat16)
        grid_thw = mx.array(processed["grid_thw"].astype(np.int32))
        patch_positions = mx.array(processed["patch_positions"].astype(np.int32))
    else:
        preprocess_s = None
        pixel_values = mx.array(fixture["pixel_values"]).astype(mx.bfloat16)
        grid_thw = mx.array(fixture["image_grid_thw"].astype(np.int32))
        patch_positions = mx.array(fixture["patch_positions"].astype(np.int32))

    runs = []
    for _ in range(args.runs):
        mx.clear_cache()
        start = time.perf_counter()
        tokens = model.generate(
            input_ids, pixel_values, grid_thw, patch_positions,
            max_new_tokens=args.max_new_tokens,
        )
        wall = time.perf_counter() - start
        runs.append({
            "wall_s": round(wall, 3),
            "tokens": len(tokens),
            "tokens_per_s": round(len(tokens) / wall, 2),
        })

    result = {
        "load_s": round(load_s, 2),
        "preprocess_s": None if preprocess_s is None else round(preprocess_s, 3),
        "prompt_tokens": int(input_ids.shape[1]),
        "runs": runs,
        "peak_memory_gb": round(mx.get_peak_memory() / 1024**3, 3),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
