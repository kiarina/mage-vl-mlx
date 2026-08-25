"""Check the MLX port against fixtures from the official PyTorch implementation.

Evaluates the Stage 1 gates: vision tower relative error and cosine, and
exact greedy token match.
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mage_vl_mlx.model import MageVL  # noqa: E402


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
    parser.add_argument("--weights", type=Path, default=Path("weights/mage-vl-bf16"))
    parser.add_argument("--fixtures", type=Path, default=Path("fixtures/stage1"))
    parser.add_argument("--device", default="mps", help="fixture device to compare against")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    compute_dtype = getattr(mx, args.dtype)
    model = MageVL.from_pretrained(args.weights)
    if compute_dtype != mx.bfloat16:
        model.update(model.apply(lambda p: p.astype(compute_dtype)))
        mx.eval(model.parameters())

    tag = f"{args.device}-{args.dtype}"
    cases = sorted(p.parent.name for p in args.fixtures.glob(f"*/{tag}.npz"))
    if not cases:
        raise SystemExit(f"no fixtures tagged {tag} under {args.fixtures}")

    report = {}
    for case in cases:
        fixture = np.load(args.fixtures / case / f"{tag}.npz")
        pixel_values = mx.array(fixture["pixel_values"]).astype(compute_dtype)
        grid_thw = mx.array(fixture["image_grid_thw"].astype(np.int32))
        patch_positions = mx.array(fixture["patch_positions"].astype(np.int32))
        input_ids = mx.array(fixture["input_ids"].astype(np.int32))

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
        matched = sum(
            1 for a, b in zip(generated, reference_ids) if a == b
        )
        report[case] = {
            "visual": visual_stats,
            "logits_last": logits_stats,
            "greedy_match": generated == reference_ids,
            "greedy_matched_tokens": f"{matched}/{len(reference_ids)}",
            "first_mismatch": next(
                (i for i, (a, b) in enumerate(zip(generated, reference_ids)) if a != b), None
            ),
        }
        v, lg = visual_stats, logits_stats
        print(f"{case}:")
        print(f"  visual      rel={v['rel_error']:.3e} cos={v['cosine']:.6f} "
              f"max_abs={v['max_abs_diff']:.3e}")
        print(f"  logits_last rel={lg['rel_error']:.3e} cos={lg['cosine']:.6f}")
        print(f"  greedy      {report[case]['greedy_matched_tokens']} "
              f"match={report[case]['greedy_match']}")

    out = args.fixtures / f"parity-{tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
