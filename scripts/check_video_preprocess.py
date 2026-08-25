"""Compare this port's torch-free video preprocessing against the official one.

This is the first Stage 2 gate: selected frame indices and preprocessed pixel
values must match the official implementation.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mage_vl_mlx.video import preprocess_video  # noqa: E402

MODEL_ID = "microsoft/Mage-VL"
REVISION = "d88b153285f1633a61b2f693c59c8576693af185"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", action="append", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("fixtures/stage2"))
    args = parser.parse_args()

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True
    )
    video_processor = processor.video_processor

    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for video in args.video:
        reference = video_processor([str(video)])
        ref_pv = reference["pixel_values_videos"].numpy()
        ref_grid = reference["video_grid_thw"].numpy()
        ref_pp = reference["patch_positions"].numpy()

        actual = preprocess_video(str(video))
        act_pv, act_grid, act_pp = (
            actual["pixel_values"], actual["grid_thw"], actual["patch_positions"]
        )

        ref_t = sorted(set(ref_pp[:, 0].tolist()))
        entry = {
            "grid_thw": {"reference": ref_grid.tolist(), "actual": act_grid.tolist(),
                         "match": ref_grid.tolist() == act_grid.tolist()},
            "frame_indices": {"reference": ref_t, "actual": actual["frame_indices"],
                              "match": ref_t == actual["frame_indices"]},
            "patch_positions_match": bool(np.array_equal(ref_pp, act_pp)),
            "pixel_values": {
                "shape_match": ref_pv.shape == act_pv.shape,
                "exact_match": bool(np.array_equal(ref_pv, act_pv)),
                "max_abs_diff": float(np.max(np.abs(ref_pv - act_pv)))
                if ref_pv.shape == act_pv.shape else None,
            },
        }
        report[video.name] = entry
        pv = entry["pixel_values"]
        print(f"{video.name}: grid={entry['grid_thw']['match']} "
              f"frames={entry['frame_indices']['match']} "
              f"positions={entry['patch_positions_match']} "
              f"pixels_exact={pv['exact_match']} max_abs={pv['max_abs_diff']}")

        stem = video.stem
        (args.out / stem).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.out / stem / "reference-preprocess.npz",
            pixel_values=ref_pv, grid_thw=ref_grid, patch_positions=ref_pp,
        )

    (args.out / "preprocess-report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out / 'preprocess-report.json'}")


if __name__ == "__main__":
    main()
