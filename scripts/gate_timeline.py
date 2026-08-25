"""Run the streaming gate over a video and print a p_speak timeline.

Reports one probability per sampled frame with its timestamp, so a gate
decision can be lined up against a hand-annotated event time.
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mage_vl_mlx.model import MageVL  # noqa: E402
from mage_vl_mlx.streaming import StreamMindGate  # noqa: E402
from mage_vl_mlx.video import preprocess_video  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--weights", type=Path, default=Path("weights/mage-vl-bf16"))
    parser.add_argument("--dtype", default="float32", choices=("bfloat16", "float32"))
    parser.add_argument("--target-fps", type=float, default=None)
    parser.add_argument("--fixed-num-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=24.0,
                        help="source fps, used to turn frame indices into seconds")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    dtype = getattr(mx, args.dtype)
    model = MageVL.from_pretrained(args.weights)
    gate = StreamMindGate()
    gate.load_weights(str(args.weights / "streammind_gate.safetensors"))
    if dtype != mx.bfloat16:
        model.update(model.apply(lambda p: p.astype(dtype)))
        gate.update(gate.apply(lambda p: p.astype(dtype)))
    mx.eval(model.parameters(), gate.parameters())

    processed = preprocess_video(
        str(args.video),
        target_fps=args.target_fps,
        fixed_num_frames=args.fixed_num_frames,
    )
    tokens = model.vision_tokens(
        mx.array(processed["pixel_values"]).astype(dtype),
        mx.array(processed["grid_thw"].astype(np.int32)),
        mx.array(processed["patch_positions"].astype(np.int32)),
    )
    logits = gate(tokens)
    mx.eval(logits)

    values = np.array(logits.astype(mx.float32))
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted.astype(np.float64))
    p_speak = (exp / exp.sum(axis=-1, keepdims=True))[0, :, 1]

    rows = []
    for index, probability in zip(processed["frame_indices"], p_speak):
        rows.append({
            "frame": int(index),
            "seconds": round(index / args.fps, 3),
            "p_speak": round(float(probability), 6),
            "decision": "speak" if probability >= args.threshold else "silent",
        })
        print(f"  frame {index:4d}  t={index / args.fps:6.2f}s  "
              f"p_speak={probability:.4f}  {rows[-1]['decision']}")

    speaks = [r for r in rows if r["decision"] == "speak"]
    print(f"frames={len(rows)}  speak={len(speaks)}  max_p_speak={p_speak.max():.4f}")
    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"video": str(args.video), "dtype": args.dtype,
             "target_fps": args.target_fps, "threshold": args.threshold,
             "timeline": rows}, indent=2))


if __name__ == "__main__":
    main()
