"""Online segment processing and latency measurement for Mage-VL.

``RealtimeSession`` accepts one completed video segment at a time. It keeps the
causal visual history used by StreamMind, emits token callbacks while text is
being decoded, and returns timing data for the complete segment pipeline.

The current implementation replays the accumulated visual history through the
gate for every new segment. This preserves the behavior of the official
whole-stream script without pretending that the MLX Mamba port has an
incremental-state API. Long-stream benchmarks should report the resulting
backlog; a future stateful gate can replace the replay behind this interface.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
import time

import mlx.core as mx
import numpy as np

from .model import MageVL
from .prompt import PromptBuilder
from .streaming import StreamMindGate

EOS_TOKEN_ID = 151645

TokenCallback = Callable[[int, str, int, float], None]


def video_duration(path: str | Path) -> float:
    """Return media duration using ffprobe."""
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(completed.stdout.strip())


def extract_subclip(
    source: str | Path,
    start_s: float,
    duration_s: float,
    output: str | Path,
) -> Path:
    """Cut an exact, independently decodable H.264 segment with ffmpeg."""
    output = Path(output)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start_s:.3f}",
            "-t", f"{duration_s:.3f}", "-i", str(source), "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-an", str(output),
        ],
        check=True,
    )
    return output


@dataclass(frozen=True)
class SegmentMetrics:
    """Wall-clock measurements for one completed input segment."""

    preprocess_s: float
    vision_s: float
    gate_s: float
    first_text_s: float | None
    first_token_s: float | None
    generation_s: float | None
    total_s: float
    generated_tokens: int
    tokens_per_s: float | None
    peak_memory_gb: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SegmentResult:
    """Gate decision, generated text, and metrics for one segment."""

    start_s: float
    end_s: float
    probability: float
    responded: bool
    text: str
    metrics: SegmentMetrics

    def to_dict(self) -> dict:
        result = asdict(self)
        result["metrics"] = self.metrics.to_dict()
        return result


class RealtimeSession:
    """Process a causal video stream one completed segment at a time."""

    def __init__(
        self,
        model: MageVL,
        gate: StreamMindGate,
        prompt_builder: PromptBuilder,
        *,
        model_dtype=mx.bfloat16,
        gate_dtype=mx.float32,
        video_backend: str = "frames",
        num_frames: int = 16,
        target_fps: float = 2.0,
        gate_threshold: float = 0.3,
        max_new_tokens: int = 80,
        codec_cache_root: str | Path | None = None,
        codec_cache_ephemeral: bool = False,
    ):
        if video_backend not in {"frames", "codec"}:
            raise ValueError(f"unsupported video backend: {video_backend}")
        self.model = model
        self.gate = gate
        self.prompt_builder = prompt_builder
        self.model_dtype = model_dtype
        self.gate_dtype = gate_dtype
        self.video_backend = video_backend
        self.num_frames = num_frames
        self.target_fps = target_fps
        self.gate_threshold = gate_threshold
        self.max_new_tokens = max_new_tokens
        # cv-preinfer caches its assets per video path and never evicts them.
        # A live stream produces a new file per segment and never revisits one,
        # so caching there only grows on disk (~570 KB per segment). Callers
        # feeding a stream should keep the cache ephemeral.
        self.codec_cache_root = Path(codec_cache_root) if codec_cache_root else None
        self.codec_cache_ephemeral = codec_cache_ephemeral
        self._vision_history: list[mx.array] = []
        self._boundaries: list[int] = []

    @classmethod
    def from_pretrained(
        cls,
        weights: str | Path,
        *,
        model_dtype=mx.bfloat16,
        gate_dtype=mx.float32,
        **kwargs,
    ) -> "RealtimeSession":
        weights = Path(weights)
        model = MageVL.from_pretrained(weights)
        gate = StreamMindGate()
        gate.load_weights(str(weights / "streammind_gate.safetensors"))
        if model_dtype != mx.bfloat16:
            model.update(model.apply(lambda value: value.astype(model_dtype)))
        if gate_dtype != mx.bfloat16:
            gate.update(gate.apply(lambda value: value.astype(gate_dtype)))
        mx.eval(model.parameters(), gate.parameters())
        return cls(
            model,
            gate,
            PromptBuilder(weights),
            model_dtype=model_dtype,
            gate_dtype=gate_dtype,
            **kwargs,
        )

    def reset(self) -> None:
        """Forget the causal gate history before starting another stream."""
        self._vision_history.clear()
        self._boundaries.clear()

    def _preprocess(self, video_path: str | Path) -> dict:
        if self.video_backend == "codec":
            import shutil

            from .codec import preprocess_codec, run_cv_preinfer

            assets = run_cv_preinfer(video_path, cache_root=self.codec_cache_root)
            try:
                return preprocess_codec(assets)
            finally:
                if self.codec_cache_ephemeral:
                    shutil.rmtree(assets, ignore_errors=True)

        from .video import preprocess_video

        return preprocess_video(
            str(video_path),
            max_frames=self.num_frames,
            target_fps=self.target_fps,
        )

    @staticmethod
    def _speak_probability(logits: mx.array, boundary: int) -> float:
        mx.eval(logits)
        picked = np.asarray(logits.astype(mx.float32))[0, boundary - 1]
        shifted = picked.astype(np.float64) - float(np.max(picked))
        values = np.exp(shifted)
        return float(values[1] / values.sum())

    def process_segment(
        self,
        video_path: str | Path,
        question: str,
        *,
        start_s: float = 0.0,
        end_s: float | None = None,
        on_token: TokenCallback | None = None,
    ) -> SegmentResult:
        """Process one completed segment and optionally stream decoded text."""
        pipeline_start = time.perf_counter()

        phase_start = time.perf_counter()
        processed = self._preprocess(video_path)
        preprocess_s = time.perf_counter() - phase_start

        pixel_values = mx.array(processed["pixel_values"]).astype(self.model_dtype)
        grid = mx.array(np.asarray(processed["grid_thw"]).astype(np.int32))
        positions = mx.array(
            np.asarray(processed["patch_positions"]).astype(np.int32)
        )

        phase_start = time.perf_counter()
        vision_tokens = self.model.vision_tokens(pixel_values, grid, positions)
        mx.eval(vision_tokens)
        vision_s = time.perf_counter() - phase_start

        self._vision_history.append(vision_tokens.astype(self.gate_dtype))
        previous = self._boundaries[-1] if self._boundaries else 0
        self._boundaries.append(previous + int(vision_tokens.shape[1]))

        phase_start = time.perf_counter()
        logits = self.gate(
            mx.concatenate(self._vision_history, axis=1),
            response_positions=self._boundaries,
        )
        probability = self._speak_probability(logits, self._boundaries[-1])
        gate_s = time.perf_counter() - phase_start

        tokens: list[int] = []
        text = ""
        first_text_s = None
        first_token_s = None
        generation_s = None
        tokens_per_s = None
        responded = probability >= self.gate_threshold
        if responded:
            if self.video_backend == "codec":
                input_ids = self.prompt_builder.for_video_codec(
                    question, processed["patch_positions"], processed["fps"]
                )
            else:
                input_ids = self.prompt_builder.for_video_frames(
                    question,
                    processed["grid_thw"],
                    processed["frame_timestamps"],
                )

            generation_start = time.perf_counter()
            for index, token in enumerate(self.model.generate_stream(
                mx.array(input_ids),
                pixel_values,
                grid,
                positions,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=EOS_TOKEN_ID,
            ), start=1):
                elapsed = time.perf_counter() - generation_start
                if first_token_s is None:
                    first_token_s = elapsed
                    first_text_s = time.perf_counter() - pipeline_start
                tokens.append(token)
                text = self.prompt_builder.decode(tokens).strip()
                if on_token is not None:
                    on_token(token, text, index, elapsed)
            generation_s = time.perf_counter() - generation_start
            if generation_s > 0:
                tokens_per_s = len(tokens) / generation_s

        total_s = time.perf_counter() - pipeline_start
        metrics = SegmentMetrics(
            preprocess_s=preprocess_s,
            vision_s=vision_s,
            gate_s=gate_s,
            first_text_s=first_text_s,
            first_token_s=first_token_s,
            generation_s=generation_s,
            total_s=total_s,
            generated_tokens=len(tokens),
            tokens_per_s=tokens_per_s,
            peak_memory_gb=mx.get_peak_memory() / 1024**3,
        )
        return SegmentResult(
            start_s=start_s,
            end_s=start_s if end_s is None else end_s,
            probability=probability,
            responded=responded,
            text=text,
            metrics=metrics,
        )
