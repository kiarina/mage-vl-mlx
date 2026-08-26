#!/usr/bin/env python
"""Run event-gated Mage-VL inference on a video, on Apple Silicon with MLX.

Mirrors microsoft/Mage's mage_vl/inference_streaming.py. The video is split
into non-overlapping segments; the StreamMind gate scores the whole causal
segment stream once, and generation runs only on segments whose score crosses
the threshold.

What the gate actually decides, measured on this port: it separates content
types (a sports broadcast scores ~0.7-0.8, a quiet hallway ~0.05-0.11) but it
does not track event times within a stream. Treat it as "is this stream worth
commentating on", not as an event detector.

The codec backend needs cv-preinfer, which has no macOS build. Point
CV_PREINFER_BIN at docker/cv-preinfer to run it through a container.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mage_vl_mlx.model import MageVL  # noqa: E402
from mage_vl_mlx.prompt import PromptBuilder  # noqa: E402
from mage_vl_mlx.realtime import extract_subclip, video_duration  # noqa: E402
from mage_vl_mlx.streaming import StreamMindGate  # noqa: E402

EOS_TOKEN_ID = 151645
USER_PROMPT = "Please describe the video content in detail based on the provided information."


def preprocess_segment(clip: Path, backend: str, num_frames: int, cur_fps: float):
    if backend == "codec":
        from mage_vl_mlx.codec import preprocess_codec, run_cv_preinfer

        return preprocess_codec(run_cv_preinfer(clip))
    from mage_vl_mlx.video import preprocess_video

    return preprocess_video(str(clip), max_frames=num_frames, target_fps=cur_fps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--weights", type=Path, default=Path("weights/mage-vl-bf16"))
    parser.add_argument("--video-backend", choices=("codec", "frames"), default="codec")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--cur-fps", type=float, default=2.0)
    parser.add_argument("--segment-sec", type=float, default=8.0)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--question", default=USER_PROMPT)
    parser.add_argument("--dtype", default="float32", choices=("bfloat16", "float32"),
                        help="float32 by default: the gate's decision flips under "
                             "bfloat16 when a score sits near the threshold")
    args = parser.parse_args()
    if args.segment_sec <= 0:
        parser.error("--segment-sec must be greater than 0")

    dtype = getattr(mx, args.dtype)
    model = MageVL.from_pretrained(args.weights)
    gate = StreamMindGate()
    gate.load_weights(str(args.weights / "streammind_gate.safetensors"))
    if dtype != mx.bfloat16:
        model.update(model.apply(lambda p: p.astype(dtype)))
        gate.update(gate.apply(lambda p: p.astype(dtype)))
    mx.eval(model.parameters(), gate.parameters())
    builder = PromptBuilder(args.weights)

    duration = video_duration(args.video)
    segments = []
    with tempfile.TemporaryDirectory(dir=str(args.video.resolve().parent)) as tmp:
        start = 0.0
        while start < duration - 1e-3:
            if args.max_segments and len(segments) >= args.max_segments:
                break
            end = min(duration, start + args.segment_sec)
            clip = extract_subclip(args.video, start, end - start,
                                   Path(tmp) / f"seg_{int(start * 1000):08d}.mp4")
            try:
                processed = preprocess_segment(
                    clip, args.video_backend, args.num_frames, args.cur_fps)
            except Exception as error:
                print(f"[t={start:.1f}-{end:.1f}s] skip (segment unusable: "
                      f"{type(error).__name__})")
                start = end
                continue
            processed["tokens"] = model.vision_tokens(
                mx.array(processed["pixel_values"]).astype(dtype),
                mx.array(processed["grid_thw"].astype(np.int32)),
                mx.array(processed["patch_positions"].astype(np.int32)))
            segments.append((start, end, processed))
            start = end

        if not segments:
            print("no usable segments")
            return

        lengths = [s[2]["tokens"].shape[1] for s in segments]
        boundaries = [int(b) for b in np.cumsum(lengths)]
        logits = gate(
            mx.concatenate([s[2]["tokens"] for s in segments], axis=1),
            response_positions=boundaries)
        mx.eval(logits)
        picked = np.array(logits.astype(mx.float32))[0][[b - 1 for b in boundaries]]
        exp = np.exp((picked - picked.max(axis=-1, keepdims=True)).astype(np.float64))
        probabilities = (exp / exp.sum(axis=-1, keepdims=True))[:, 1]

        for (start, end, processed), probability in zip(segments, probabilities):
            if probability < args.gate_threshold:
                print(f"[t={start:.1f}-{end:.1f}s] gate=silence (p={probability:.2f})")
                continue
            if args.video_backend == "codec":
                ids = builder.for_video_codec(
                    args.question, processed["patch_positions"], processed["fps"])
            else:
                ids = builder.for_video_frames(
                    args.question, processed["grid_thw"], processed["frame_timestamps"])
            tokens = model.generate(
                mx.array(ids),
                mx.array(processed["pixel_values"]).astype(dtype),
                mx.array(processed["grid_thw"].astype(np.int32)),
                mx.array(processed["patch_positions"].astype(np.int32)),
                max_new_tokens=args.max_new_tokens, eos_token_id=EOS_TOKEN_ID)
            text = builder.decode(tokens).strip()
            print(f"[t={start:.1f}-{end:.1f}s] gate=response (p={probability:.2f}) -> {text}")


if __name__ == "__main__":
    main()
