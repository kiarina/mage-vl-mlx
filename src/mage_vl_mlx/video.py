"""Torch-free frame sampling and video preprocessing for Mage-VL.

Mirrors video_processing_mage_vl.py's OpenCV path and the Qwen2VL patchify
it delegates to, using only OpenCV, NumPy, and PIL.

Note on the second resize: extract_frames aligns to patch_size * 2 (32) with
a 200704..1605632 pixel budget, and the checkpoint's image processor then
applies its own smart_resize with the same factor but a far wider budget
(3136..4000000). The frames are therefore already aligned and in range, so
that second resize is a no-op and no torchvision interpolation is needed.
"""

from __future__ import annotations

import math

import numpy as np

IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
IMAGE_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
RESCALE = 1.0 / 255.0


def format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds - minutes * 60:09.6f}"


def choose_target_frames(
    duration_seconds: float,
    max_frames: int,
    fixed_num_frames: int | None = None,
    target_fps: float | None = None,
) -> int:
    if target_fps is not None and target_fps > 0:
        return min(max(1, int(duration_seconds * target_fps)), max_frames)
    if fixed_num_frames is not None:
        return fixed_num_frames
    if duration_seconds < 10:
        return 8
    if duration_seconds < 30:
        return 16
    return max_frames


def select_frame_indices(frame_count: int, target_count: int) -> list[int]:
    """Evenly spaced indices. numpy and torch both round half to even."""
    if frame_count <= target_count:
        return list(range(frame_count))
    return np.rint(np.linspace(0, frame_count - 1, target_count)).astype(np.int64).tolist()


def smart_resize(
    height: int,
    width: int,
    patch_size: int = 16,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    align_patch_size: int | None = None,
) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid size: height={height}, width={width}")
    factor = align_patch_size or patch_size
    h_bar = max(factor, int(round(height / factor) * factor))
    w_bar = max(factor, int(round(width / factor) * factor))
    if max_pixels and h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif min_pixels and h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return int(h_bar), int(w_bar)


def extract_frames(
    video_path: str,
    max_frames: int = 384,
    patch_size: int = 16,
    min_pixels: int = 200704,
    max_pixels: int = 1605632,
    resize_frames: bool = True,
    fixed_num_frames: int | None = None,
    target_fps: float | None = None,
) -> tuple[list[np.ndarray], list[int], list[float]]:
    """Decode and sample frames. Returns (RGB uint8 frames, indices, seconds)."""
    import cv2

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        capture.release()
        raise ValueError(f"unknown frame count for video: {video_path}")

    target = choose_target_frames(
        frame_count / fps, max_frames, fixed_num_frames, target_fps
    )
    frames, indices, seconds = [], [], []
    for index in select_frame_indices(frame_count, target):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize_frames and (min_pixels or max_pixels):
            height, width = frame.shape[:2]
            new_h, new_w = smart_resize(
                height, width, patch_size, min_pixels, max_pixels,
                align_patch_size=patch_size * 2,
            )
            if (new_h, new_w) != (height, width):
                interp = (
                    cv2.INTER_AREA if new_h < height or new_w < width
                    else cv2.INTER_LINEAR
                )
                frame = cv2.resize(frame, (new_w, new_h), interpolation=interp)
        frames.append(frame)
        indices.append(int(index))
        # Timestamps round-trip through "MM:SS.xx" exactly as the reference does.
        seconds.append(round(index / fps, 6))
    capture.release()
    if not frames:
        raise ValueError(f"no frames decoded from video: {video_path}")
    return frames, indices, seconds


def patchify(frames: list[np.ndarray], patch_size: int = 16, merge_size: int = 2):
    """Normalize frames and flatten to Qwen2VL patch rows.

    Returns (pixel_values [T*gh*gw, C*P*P], grid_h, grid_w).
    """
    # transformers fuses the rescale factor into mean/std and normalizes the
    # unscaled 0..255 values, so match that order to stay bit-exact.
    mean = (IMAGE_MEAN.astype(np.float64) / RESCALE).astype(np.float32)
    std = (IMAGE_STD.astype(np.float64) / RESCALE).astype(np.float32)
    stacked = (np.stack(frames).astype(np.float32) - mean) / std  # [T, H, W, C]
    images = stacked.transpose(0, 3, 1, 2)  # [T, C, H, W]

    batch, channels, height, width = images.shape
    grid_h, grid_w = height // patch_size, width // patch_size
    patches = images.reshape(
        batch, channels,
        grid_h // merge_size, merge_size, patch_size,
        grid_w // merge_size, merge_size, patch_size,
    )
    patches = patches.transpose(0, 2, 5, 3, 6, 1, 4, 7)
    flat = patches.reshape(batch * grid_h * grid_w, channels * patch_size * patch_size)
    return np.ascontiguousarray(flat), grid_h, grid_w


def build_patch_positions(
    frame_indices: list[int], grid_h: int, grid_w: int, merge_size: int = 2
) -> np.ndarray:
    """Block-layout [t, h, w] positions. The t axis uses real frame indices."""
    t = len(frame_indices)
    h_coords = np.tile(np.repeat(np.arange(grid_h), grid_w), t)
    w_coords = np.tile(np.arange(grid_w), grid_h * t)
    t_coords = np.repeat(np.asarray(frame_indices, dtype=np.int64), grid_h * grid_w)
    positions = np.stack([t_coords, h_coords, w_coords], axis=1)

    order = (
        np.arange(t * grid_h * grid_w)
        .reshape(t, grid_h // merge_size, merge_size, grid_w // merge_size, merge_size)
        .transpose(0, 1, 3, 2, 4)
        .reshape(-1)
    )
    return positions[order]


def preprocess_video(
    video_path: str,
    max_frames: int = 384,
    patch_size: int = 16,
    merge_size: int = 2,
    min_pixels: int = 200704,
    max_pixels: int = 1605632,
    fixed_num_frames: int | None = None,
    target_fps: float | None = None,
) -> dict:
    """Full torch-free video preprocessing for the frame-sampled path."""
    frames, indices, seconds = extract_frames(
        video_path, max_frames=max_frames, patch_size=patch_size,
        min_pixels=min_pixels, max_pixels=max_pixels,
        fixed_num_frames=fixed_num_frames, target_fps=target_fps,
    )
    pixel_values, grid_h, grid_w = patchify(frames, patch_size, merge_size)
    # MageVLProcessor routes videos through the image path and emits one grid
    # row per frame. That is what the model sees during official inference, and
    # it matters: with t=1 per row the vision tower attends within each frame,
    # whereas the merged [T, h, w] form would attend across frame_windows_size
    # frames. Match the inference path and expose the merged form separately.
    return {
        "pixel_values": pixel_values,
        "grid_thw": np.array([[1, grid_h, grid_w]] * len(frames), dtype=np.int64),
        "video_grid_thw": np.array([[len(frames), grid_h, grid_w]], dtype=np.int64),
        "patch_positions": build_patch_positions(indices, grid_h, grid_w, merge_size),
        "frame_indices": indices,
        "frame_timestamps": seconds,
    }
