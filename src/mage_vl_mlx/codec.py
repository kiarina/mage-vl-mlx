"""Torch-free consumption of Mage-VL codec asset directories.

cv-preinfer (codec-video-prep) selects the informative patches and packs them
into canvases; this module turns that asset directory into the tensors the
model consumes. Patch selection itself happens in cv-preinfer, so it is
identical to the reference by construction — what this reimplements is the
consumption side that the official processor performs.

codec-video-prep ships Linux-only wheels. Generate the assets in a container
(see docker/) and point this at the resulting directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .video import patchify


def load_codec_assets(asset_dir: str | Path) -> dict:
    """Read canvases, source patch positions, and metadata from an asset dir."""
    asset_dir = Path(asset_dir)
    meta = json.loads((asset_dir / "meta.json").read_text())

    canvas_files = meta.get("canvas_files")
    if not canvas_files:
        for ext in ("npy", "jpg", "png"):
            hits = sorted(p.name for p in asset_dir.glob(f"canvas_*.{ext}"))
            if hits:
                canvas_files = hits
                break
    if not canvas_files:
        raise ValueError(f"no canvases found in {asset_dir}")

    canvases = []
    for name in canvas_files:
        path = asset_dir / name
        if name.endswith(".npy"):
            canvases.append(np.load(path))
        else:
            canvases.append(np.asarray(Image.open(path).convert("RGB")))

    return {
        "canvases": canvases,
        "src_positions": np.load(asset_dir / "src_patch_position.npy"),
        "fps": float(meta.get("fps") or 30.0),
        "meta": meta,
    }


def drop_padding_canvases(
    canvases: list[np.ndarray], src_positions: np.ndarray
) -> tuple[list[np.ndarray], np.ndarray, int]:
    """Drop canvases that are entirely padding, marked by negative timestamps."""
    count = len(canvases)
    if count == 0:
        return canvases, src_positions, 0
    if src_positions.shape[0] % count != 0:
        raise ValueError(
            f"src_positions length {src_positions.shape[0]} "
            f"not divisible by canvas count {count}"
        )
    per_canvas = src_positions.shape[0] // count
    positions = src_positions.reshape(count, per_canvas, 3)
    times = positions[..., 0]

    keep = (times >= 0).any(axis=1)
    if bool((keep & ~(times >= 0).all(axis=1)).any()):
        raise ValueError("half-padding canvas; padding is expected to be canvas-granular")
    dropped = int(count - int(keep.sum()))
    if dropped == 0:
        return canvases, src_positions, 0
    kept = [c for c, flag in zip(canvases, keep.tolist()) if flag]
    return kept, positions[keep].reshape(-1, 3), dropped


def to_block_layout(
    positions: np.ndarray, grid_h: int, grid_w: int, merge_size: int = 2
) -> np.ndarray:
    """Reorder one canvas's row-major positions into 2x2 block order."""
    order = (
        np.arange(grid_h * grid_w)
        .reshape(1, grid_h // merge_size, merge_size, grid_w // merge_size, merge_size)
        .transpose(0, 1, 3, 2, 4)
        .reshape(-1)
    )
    return positions[order]


def preprocess_codec(
    asset_dir: str | Path, patch_size: int = 16, merge_size: int = 2
) -> dict:
    """Codec asset directory -> pixel_values, grid_thw, patch_positions."""
    assets = load_codec_assets(asset_dir)
    canvases, src_positions, dropped = drop_padding_canvases(
        assets["canvases"], assets["src_positions"]
    )

    heights = {c.shape[0] for c in canvases}
    widths = {c.shape[1] for c in canvases}
    if len(heights) != 1 or len(widths) != 1:
        raise ValueError(f"canvases differ in size: {heights} x {widths}")

    # Canvases are already aligned to the patch grid, so the image processor's
    # smart_resize is a no-op here and patchify can run directly.
    pixel_values, grid_h, grid_w = patchify(canvases, patch_size, merge_size)

    per_canvas = grid_h * grid_w
    if src_positions.shape[0] != per_canvas * len(canvases):
        raise ValueError(
            f"position count {src_positions.shape[0]} != "
            f"{per_canvas} x {len(canvases)} canvases"
        )
    blocks = [
        to_block_layout(
            src_positions[i * per_canvas:(i + 1) * per_canvas], grid_h, grid_w, merge_size
        )
        for i in range(len(canvases))
    ]

    return {
        "pixel_values": pixel_values,
        "grid_thw": np.array([[1, grid_h, grid_w]] * len(canvases), dtype=np.int64),
        "patch_positions": np.concatenate(blocks, axis=0).astype(np.int64),
        "canvas_count": len(canvases),
        "dropped_canvases": dropped,
        "fps": assets["fps"],
    }
