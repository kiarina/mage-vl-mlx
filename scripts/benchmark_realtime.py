"""Measure Mage-VL segment latency and simulated live-stream backlog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import mlx.core as mx

from mage_vl_mlx.realtime import (
    RealtimeSession,
    extract_subclip,
    video_duration,
)


def dtype(name: str):
    return getattr(mx, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--weights", type=Path, default=Path("weights/mage-vl-bf16"))
    parser.add_argument("--backend", choices=("frames", "codec"), default="frames")
    parser.add_argument("--segment-sec", type=float, default=4.0)
    parser.add_argument("--target-fps", type=float, default=2.0)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--gate-threshold", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--question", default="Describe what is happening. Focus on changes and motion.")
    parser.add_argument("--model-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--gate-dtype", choices=("bfloat16", "float32"), default="float32")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--min-tail-sec", type=float, default=0.5,
                        help="ignore a final container-duration sliver shorter than this")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.segment_sec <= 0:
        parser.error("--segment-sec must be greater than zero")

    load_start = time.perf_counter()
    template = RealtimeSession.from_pretrained(
        args.weights,
        model_dtype=dtype(args.model_dtype),
        gate_dtype=dtype(args.gate_dtype),
        video_backend=args.backend,
        num_frames=args.num_frames,
        target_fps=args.target_fps,
        gate_threshold=args.gate_threshold,
        max_new_tokens=args.max_new_tokens,
    )
    load_s = time.perf_counter() - load_start
    duration = video_duration(args.video)
    runs = []

    for run_index in range(1, args.runs + 1):
        session = RealtimeSession(
            template.model,
            template.gate,
            template.prompt_builder,
            model_dtype=dtype(args.model_dtype),
            gate_dtype=dtype(args.gate_dtype),
            video_backend=args.backend,
            num_frames=args.num_frames,
            target_fps=args.target_fps,
            gate_threshold=args.gate_threshold,
            max_new_tokens=args.max_new_tokens,
        )
        segments = []
        worker_free_s = 0.0
        processing_s = 0.0
        segment_work_fits = []
        with tempfile.TemporaryDirectory(dir=args.video.resolve().parent) as directory:
            start_s = 0.0
            segment_index = 0
            while start_s < duration - 1e-3:
                if start_s > 0 and duration - start_s < args.min_tail_sec:
                    break
                end_s = min(duration, start_s + args.segment_sec)
                segment_index += 1
                clip = Path(directory) / f"segment-{segment_index:04d}.mp4"
                prepare_start = time.perf_counter()
                extract_subclip(args.video, start_s, end_s - start_s, clip)
                prepare_s = time.perf_counter() - prepare_start
                backlog_before_s = max(0.0, worker_free_s - end_s)
                result = session.process_segment(
                    clip,
                    args.question,
                    start_s=start_s,
                    end_s=end_s,
                )
                work_s = prepare_s + result.metrics.total_s
                segment_work_fits.append(work_s <= end_s - start_s)
                worker_free_s = max(worker_free_s, end_s) + work_s
                processing_s += work_s
                before_generation_s = (
                    prepare_s
                    + result.metrics.preprocess_s
                    + result.metrics.vision_s
                    + result.metrics.gate_s
                )
                first_text_s = None
                if result.metrics.first_token_s is not None:
                    first_text_s = (
                        backlog_before_s
                        + before_generation_s
                        + result.metrics.first_token_s
                    )
                segments.append({
                    "index": segment_index,
                    "start_s": start_s,
                    "end_s": end_s,
                    "prepare_s": prepare_s,
                    "work_s": work_s,
                    "backlog_before_s": backlog_before_s,
                    "first_text_after_boundary_s": first_text_s,
                    "full_response_after_boundary_s": backlog_before_s + work_s,
                    "lag_after_s": max(0.0, worker_free_s - end_s),
                    "result": result.to_dict(),
                })
                start_s = end_s
        runs.append({
            "run": run_index,
            "processing_s": processing_s,
            "real_time_factor": processing_s / duration,
            "tail_response_s": max(0.0, worker_free_s - duration),
            "max_backlog_before_s": max(
                (segment["backlog_before_s"] for segment in segments), default=0.0
            ),
            "each_segment_fits_interval": all(segment_work_fits),
            "segments": segments,
        })

    report = {
        "video": str(args.video),
        "video_duration_s": duration,
        "weights": str(args.weights),
        "backend": args.backend,
        "segment_s": args.segment_sec,
        "target_fps": args.target_fps,
        "num_frames": args.num_frames,
        "gate_threshold": args.gate_threshold,
        "max_new_tokens": args.max_new_tokens,
        "model_dtype": args.model_dtype,
        "gate_dtype": args.gate_dtype,
        "question": args.question,
        "min_tail_sec": args.min_tail_sec,
        "model_load_s": load_s,
        "peak_memory_gb": mx.get_peak_memory() / 1024**3,
        "runs": runs,
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
