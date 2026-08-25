"""Check the MLX port's frame-sampled video path against official fixtures.

Runs the port's own torch-free preprocessing from the video file, so this
exercises the full Stage 2 path rather than replaying fixture tensors.
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mage_vl_mlx.model import MageVL  # noqa: E402
from mage_vl_mlx.video import preprocess_video  # noqa: E402


def stats(reference: np.ndarray, actual: np.ndarray) -> dict:
    ref, act = reference.astype(np.float64).ravel(), actual.astype(np.float64).ravel()
    denom = np.linalg.norm(ref)
    return {
        "max_abs_diff": float(np.max(np.abs(ref - act))),
        "rel_error": float(np.linalg.norm(ref - act) / denom),
        "cosine": float(np.dot(ref, act) / (denom * np.linalg.norm(act))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", action="append", required=True, type=Path)
    parser.add_argument("--weights", type=Path, default=Path("weights/mage-vl-bf16"))
    parser.add_argument("--fixtures", type=Path, default=Path("fixtures/stage2"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=("bfloat16", "float32"))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    compute_dtype = getattr(mx, args.dtype)
    model = MageVL.from_pretrained(args.weights)
    if compute_dtype != mx.bfloat16:
        model.update(model.apply(lambda p: p.astype(compute_dtype)))
        mx.eval(model.parameters())

    tag = f"{args.device}-{args.dtype}"
    report = {}
    for video in args.video:
        fixture_path = args.fixtures / video.stem / f"{tag}.npz"
        if not fixture_path.exists():
            raise SystemExit(f"missing fixture: {fixture_path}")
        fixture = np.load(fixture_path)

        processed = preprocess_video(str(video))
        pixel_values = mx.array(processed["pixel_values"]).astype(compute_dtype)
        grid_thw = mx.array(processed["grid_thw"].astype(np.int32))
        patch_positions = mx.array(processed["patch_positions"].astype(np.int32))
        input_ids = mx.array(fixture["input_ids"].astype(np.int32))

        preprocess_exact = all((
            np.array_equal(fixture["pixel_values"], processed["pixel_values"]),
            np.array_equal(
                fixture["patch_positions"].astype(np.int64),
                processed["patch_positions"],
            ),
            np.array_equal(
                fixture["image_grid_thw"].astype(np.int64), processed["grid_thw"]
            ),
        ))

        visual = model.vision(pixel_values, grid_thw, patch_positions)
        mx.eval(visual)
        visual_stats = stats(fixture["visual"], np.array(visual.astype(mx.float32)))

        embeds = model.embed(input_ids, pixel_values, grid_thw, patch_positions)
        logits = model(embeds)[:, -1]
        mx.eval(logits)
        logits_stats = stats(fixture["logits_last"], np.array(logits[0].astype(mx.float32)))

        generated = model.generate(
            input_ids, pixel_values, grid_thw, patch_positions,
            max_new_tokens=args.max_new_tokens,
        )
        reference_ids = fixture["greedy_ids"].astype(np.int64).tolist()
        matched = sum(1 for a, b in zip(generated, reference_ids) if a == b)

        report[video.stem] = {
            "preprocess_exact": preprocess_exact,
            "frames": len(processed["frame_indices"]),
            "grid_thw": processed["grid_thw"].tolist(),
            "visual": visual_stats,
            "logits_last": logits_stats,
            "greedy_match": generated == reference_ids,
            "greedy_matched_tokens": f"{matched}/{len(reference_ids)}",
        }
        entry = report[video.stem]
        print(f"{video.stem}:")
        print(f"  preprocess  exact={preprocess_exact} frames={entry['frames']} "
              f"grid={entry['grid_thw']}")
        print(f"  visual      rel={visual_stats['rel_error']:.3e} "
              f"cos={visual_stats['cosine']:.6f}")
        print(f"  logits_last rel={logits_stats['rel_error']:.3e} "
              f"cos={logits_stats['cosine']:.6f}")
        print(f"  greedy      {entry['greedy_matched_tokens']} "
              f"match={entry['greedy_match']}")

    out = args.fixtures / f"parity-{tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
