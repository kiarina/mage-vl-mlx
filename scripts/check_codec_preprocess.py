"""Compare this port's torch-free codec consumption against the official one.

Patch selection happens inside cv-preinfer, so both sides read the same asset
directory; what this checks is the consumption side — canvases to pixel values,
source positions to block-layout patch positions.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mage_vl_mlx.codec import preprocess_codec  # noqa: E402

MODEL_ID = "microsoft/Mage-VL"
REVISION = "d88b153285f1633a61b2f693c59c8576693af185"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", action="append", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("fixtures/stage4"))
    args = parser.parse_args()

    from transformers import AutoProcessor
    from codec_video_processing_mage_vl import CodecConfig, _cache_dir_for  # type: ignore

    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True
    )
    messages = [{"role": "user", "content": [
        {"type": "video"}, {"type": "text", "text": "Describe this video."},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for video in args.video:
        path = str(video.resolve())
        reference = processor(
            text=[text], videos=[path], video_backend="codec",
            codec_config={"patch": 16, "max_pixels": 150000}, max_pixels=150000,
            return_tensors="pt", padding=True,
        )
        cfg = CodecConfig(patch=16, max_pixels=150000)
        asset_dir = _cache_dir_for(path, cfg)

        actual = preprocess_codec(asset_dir)
        ref_pv = reference["pixel_values"].numpy()
        ref_pp = reference["patch_positions"].numpy()
        ref_grid = reference["image_grid_thw"].numpy()

        entry = {
            "asset_dir": str(asset_dir),
            "canvases": actual["canvas_count"],
            "dropped_canvases": actual["dropped_canvases"],
            "grid_match": ref_grid.tolist() == actual["grid_thw"].tolist(),
            "patch_positions_match": bool(np.array_equal(ref_pp, actual["patch_positions"])),
            "pixel_values_exact": bool(np.array_equal(ref_pv, actual["pixel_values"])),
            "pixel_values_max_abs_diff": float(np.max(np.abs(ref_pv - actual["pixel_values"])))
            if ref_pv.shape == actual["pixel_values"].shape else None,
            "visual_tokens": int(ref_grid.prod(-1).sum()) // 4,
        }
        report[video.stem] = entry
        print(f"{video.stem}: canvases={entry['canvases']} "
              f"grid={entry['grid_match']} positions={entry['patch_positions_match']} "
              f"pixels_exact={entry['pixel_values_exact']} "
              f"max_abs={entry['pixel_values_max_abs_diff']}")

    (args.out / "codec-preprocess-report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out / 'codec-preprocess-report.json'}")


if __name__ == "__main__":
    main()
