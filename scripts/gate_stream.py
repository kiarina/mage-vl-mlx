"""Run the streaming gate over a video the way official inference does.

inference_streaming.py splits the video into non-overlapping segments,
preprocesses each independently, concatenates their vision tokens into one
causal stream, and reads the gate only at each segment boundary with
response_positions set. A per-frame reading of the gate is not how the
checkpoint is meant to be used and gives different numbers.

Deviation from the official script: it cuts subclips with `ffmpeg -c copy`,
which snaps to keyframes. Videos with sparse keyframes cannot be cut that way
at short segment lengths, so subclips are re-encoded here.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mage_vl_mlx.model import MageVL  # noqa: E402
from mage_vl_mlx.streaming import StreamMindGate  # noqa: E402
from mage_vl_mlx.codec import preprocess_codec  # noqa: E402
from mage_vl_mlx.video import preprocess_video  # noqa: E402


def preprocess_codec_clip(clip: Path) -> dict:
    """Run cv-preinfer on one subclip, then consume the asset directory.

    Imports the checkpoint's own CodecConfig so the cache key matches what the
    official processor would compute for the same clip.
    """
    from codec_video_processing_mage_vl import (  # type: ignore
        CodecConfig, process_codec_video,
    )

    cfg = CodecConfig(patch=16, max_pixels=150000)
    payload = process_codec_video(str(clip.resolve()), cfg)
    return preprocess_codec(payload["out_dir"])


def video_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def extract_subclip(src: Path, start: float, duration: float, out_path: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}",
         "-t", f"{duration:.3f}", "-i", str(src),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(out_path)],
        check=True,
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--weights", type=Path, default=Path("weights/mage-vl-bf16"))
    parser.add_argument("--dtype", default="float32", choices=("bfloat16", "float32"))
    parser.add_argument("--segment-sec", type=float, default=8.0)
    parser.add_argument("--num-frames", type=int, default=16,
                        help="per-segment frame cap (official default 16)")
    parser.add_argument("--cur-fps", type=float, default=2.0,
                        help="per-segment sampling fps (official default 2)")
    parser.add_argument("--backend", default="frames", choices=("frames", "codec"),
                        help="official inference_streaming.py defaults to codec")
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

    duration = video_duration(args.video)
    spans, tokens = [], []
    # Subclips go next to the source, not into the system temp: on macOS that
    # lives under /var/folders, which Docker does not share, so the codec
    # wrapper would bind-mount an empty directory and see no video.
    with tempfile.TemporaryDirectory(dir=str(args.video.resolve().parent)) as tmp:
        start = 0.0
        while start < duration - 1e-3:
            end = min(duration, start + args.segment_sec)
            clip = extract_subclip(
                args.video, start, end - start,
                Path(tmp) / f"seg_{int(start * 1000):08d}.mp4",
            )
            # A trailing sliver of a segment can be too short to preprocess —
            # the codec pipeline needs enough frames to form a group. The
            # official script skips such segments, so do the same.
            try:
                if args.backend == "codec":
                    processed = preprocess_codec_clip(clip)
                    units = processed["canvas_count"]
                else:
                    processed = preprocess_video(
                        str(clip), max_frames=args.num_frames, target_fps=args.cur_fps
                    )
                    units = len(processed["frame_indices"])
            except Exception as error:
                print(f"  [t={start:5.2f}-{end:5.2f}s] skip (unusable: "
                      f"{type(error).__name__})")
                start = end
                continue
            tokens.append(model.vision_tokens(
                mx.array(processed["pixel_values"]).astype(dtype),
                mx.array(processed["grid_thw"].astype(np.int32)),
                mx.array(processed["patch_positions"].astype(np.int32)),
            ))
            spans.append((start, end, units))
            start = end

    lengths = [t.shape[1] for t in tokens]
    boundaries = [int(b) for b in np.cumsum(lengths)]
    stream = mx.concatenate(tokens, axis=1)
    logits = gate(stream, response_positions=boundaries)
    mx.eval(logits)

    values = np.array(logits.astype(mx.float32))[0]
    picked = values[[b - 1 for b in boundaries]]
    shifted = picked - picked.max(axis=-1, keepdims=True)
    exp = np.exp(shifted.astype(np.float64))
    probabilities = (exp / exp.sum(axis=-1, keepdims=True))[:, 1]

    rows = []
    for (start, end, frames), probability in zip(spans, probabilities):
        decision = "speak" if probability >= args.threshold else "silent"
        rows.append({"start": round(start, 2), "end": round(end, 2),
                     "frames": frames, "p_speak": round(float(probability), 6),
                     "decision": decision})
        print(f"  [t={start:5.2f}-{end:5.2f}s] frames={frames:3d} "
              f"p_speak={probability:.4f}  gate={decision}")

    speaks = sum(1 for r in rows if r["decision"] == "speak")
    print(f"segments={len(rows)}  speak={speaks}  "
          f"max_p_speak={probabilities.max():.4f}")
    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"video": str(args.video), "dtype": args.dtype,
             "segment_sec": args.segment_sec, "num_frames": args.num_frames,
             "cur_fps": args.cur_fps, "threshold": args.threshold,
             "segments": rows}, indent=2))


if __name__ == "__main__":
    main()
