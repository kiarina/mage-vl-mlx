"""Torch-free prompt building: chat template, vision placeholders, tokenizer.

Mirrors MageVLProcessor's text side. The chat template emits one
``<|vision_start|><|image_pad|><|vision_end|>`` block per image and a
``<|video_pad|>`` block per video; the processor then expands those into the
exact number of visual token slots the model will consume, tagging each frame
or canvas with its timestamp. Getting the count wrong shifts every visual
token, so `scripts/check_prompt.py` compares the ids against the official
processor.

Uses the `tokenizers` and `jinja2` packages — neither pulls in torch.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
VIDEO_PAD = "<|video_pad|>"

VIDEO_BLOCK_RE = re.compile(
    re.escape(VISION_START) + r"\s*" + re.escape(VIDEO_PAD) + r"\s*" + re.escape(VISION_END)
)


def seconds_tag(seconds: float, decimals: int = 1) -> str:
    return f"<{float(seconds):.{decimals}f} seconds>"


class PromptBuilder:
    """Builds model-ready token ids from messages plus visual metadata."""

    def __init__(self, checkpoint_dir: str | Path, spatial_merge_size: int = 2):
        from jinja2 import Environment
        from tokenizers import Tokenizer

        path = Path(checkpoint_dir)
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        template = (path / "chat_template.jinja").read_text()
        # keep_trailing_newline matters: the template's final newline is part
        # of the prompt the model was trained on.
        self.template = Environment(keep_trailing_newline=True).from_string(template)
        self.merge_factor = spatial_merge_size**2

    def render(self, messages: list[dict], add_generation_prompt: bool = True) -> str:
        return self.template.render(
            messages=messages, add_generation_prompt=add_generation_prompt
        )

    def expand_image_pads(self, text: str, grid_thw: np.ndarray) -> str:
        """Expand each image block to prod(grid) / merge_factor slots."""
        counts = [
            int(row[0]) * int(row[1]) * int(row[2]) // self.merge_factor
            for row in np.asarray(grid_thw)
        ]
        index = 0
        while IMAGE_PAD in text and index < len(counts):
            text = text.replace(IMAGE_PAD, "<|placeholder|>" * counts[index], 1)
            index += 1
        return text.replace("<|placeholder|>", IMAGE_PAD)

    def expand_video_frames(
        self, text: str, grid_thw: np.ndarray, frame_seconds: list[float]
    ) -> str:
        """Replace the video block with one timestamped block per frame."""
        grid = np.asarray(grid_thw)
        per_frame = int(grid[0][1]) * int(grid[0][2]) // self.merge_factor
        parts = []
        for seconds in frame_seconds:
            parts += [seconds_tag(seconds), VISION_START,
                      IMAGE_PAD * per_frame, VISION_END]
        return self._replace_video_block(text, "".join(parts))

    def expand_codec(
        self, text: str, patch_positions: np.ndarray, fps: float, decimals: int = 1
    ) -> str:
        """Replace the video block with one block per codec timestamp run."""
        times = np.asarray(patch_positions)[:, 0]
        # Runs of equal consecutive timestamps, as torch.unique_consecutive does.
        boundaries = np.flatnonzero(np.diff(times)) + 1
        starts = np.concatenate([[0], boundaries])
        counts = np.diff(np.concatenate([starts, [len(times)]]))

        parts = []
        for start, count in zip(starts, counts):
            value = int(times[start])
            slots = int(count) // self.merge_factor
            if value < 0 or slots <= 0:
                continue
            parts += [seconds_tag(value / float(fps), decimals), VISION_START,
                      IMAGE_PAD * slots, VISION_END, "\n"]
        return self._replace_video_block(text, "".join(parts))

    @staticmethod
    def _replace_video_block(text: str, replacement: str) -> str:
        match = VIDEO_BLOCK_RE.search(text)
        if match is None:
            return text
        tail = match.end()
        if tail < len(text) and text[tail] == "\n":
            tail += 1
        return text[:match.start()] + replacement + text[tail:]

    def encode(self, text: str) -> np.ndarray:
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        return np.array([ids], dtype=np.int32)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    # ------------------------------------------------------------------ api

    def for_image(self, question: str, grid_thw: np.ndarray) -> np.ndarray:
        text = self.render([{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": question}]}])
        return self.encode(self.expand_image_pads(text, grid_thw))

    def for_video_frames(
        self, question: str, grid_thw: np.ndarray, frame_seconds: list[float]
    ) -> np.ndarray:
        text = self.render([{"role": "user", "content": [
            {"type": "video"}, {"type": "text", "text": question}]}])
        return self.encode(self.expand_video_frames(text, grid_thw, frame_seconds))

    def for_video_codec(
        self, question: str, patch_positions: np.ndarray, fps: float
    ) -> np.ndarray:
        text = self.render([{"role": "user", "content": [
            {"type": "video"}, {"type": "text", "text": question}]}])
        return self.encode(self.expand_codec(text, patch_positions, fps))
