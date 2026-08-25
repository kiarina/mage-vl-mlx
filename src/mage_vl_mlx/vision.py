"""Mage-ViT vision tower in MLX.

Mirrors modeling_mage_vl.py: 3D (T,H,W) rotary embeddings with a 4:6:6 split
applied via an interleaved rotate_half, block-diagonal attention driven by
cu_seqlens, and a 2x2 spatial patch merger.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .config import VisionConfig


def rotate_half(x: mx.array) -> mx.array:
    """Interleaved rotation: (x1, x2, x3, x4) -> (-x2, x1, -x4, x3)."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    stacked = mx.stack([-x_odd, x_even], axis=-1)
    return stacked.reshape(*x.shape[:-1], x.shape[-1])


class VisionRotaryEmbedding:
    """3D rotary frequencies with a 4:6:6 T:H:W split of head_dim // 2."""

    def __init__(self, config: VisionConfig):
        half = config.head_dim // 2
        if half % 16 != 0:
            raise ValueError("head_dim // 2 must be divisible by 16 for the 4:6:6 split")
        unit = half // 16
        self.t_size, self.h_size, self.w_size = 4 * unit, 6 * unit, 6 * unit
        base = config.rope_theta
        self.inv_freq_t = 1.0 / (base ** (mx.arange(self.t_size, dtype=mx.float32) / self.t_size))
        self.inv_freq_h = 1.0 / (base ** (mx.arange(self.h_size, dtype=mx.float32) / self.h_size))
        self.inv_freq_w = 1.0 / (base ** (mx.arange(self.w_size, dtype=mx.float32) / self.w_size))

    def from_positions(self, patch_positions: mx.array) -> mx.array:
        """patch_positions: [seq, 3] of (t, h, w) -> freqs [seq, head_dim // 2]."""
        pos = patch_positions.astype(mx.float32)
        ft = mx.outer(pos[:, 0], self.inv_freq_t)
        fh = mx.outer(pos[:, 1], self.inv_freq_h)
        fw = mx.outer(pos[:, 2], self.inv_freq_w)
        return mx.concatenate([ft, fh, fw], axis=-1)


class Attention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array, cu_seqlens: list[int]) -> mx.array:
        b, seq, _ = x.shape
        qkv = self.qkv(x).reshape(b, seq, 3, self.num_heads, self.head_dim)
        q, k, v = [qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3)]

        dtype = q.dtype
        q32, k32 = q.astype(mx.float32), k.astype(mx.float32)
        q = (q32 * cos + rotate_half(q32) * sin).astype(dtype)
        k = (k32 * cos + rotate_half(k32) * sin).astype(dtype)

        # Block-diagonal attention: each cu_seqlens span attends only to itself.
        chunks = []
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            chunks.append(
                mx.fast.scaled_dot_product_attention(
                    q[:, :, start:end], k[:, :, start:end], v[:, :, start:end],
                    scale=self.scale, mask=None,
                )
            )
        out = mx.concatenate(chunks, axis=2) if len(chunks) > 1 else chunks[0]
        out = out.transpose(0, 2, 1, 3).reshape(b, seq, -1)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        # hidden_act is "gelu": exact erf GELU, not the tanh approximation.
        return self.fc2(nn.gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(config)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array, cu_seqlens: list[int]) -> mx.array:
        x = x + self.self_attn(self.layer_norm1(x), cos, sin, cu_seqlens)
        return x + self.mlp(self.layer_norm2(x))


class PatchMerger(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        merged = config.hidden_size * config.spatial_merge_size**2
        self.ln_q = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.fc1 = nn.Linear(merged, merged, bias=True)
        self.fc2 = nn.Linear(merged, config.out_hidden_size, bias=True)
        self.merged_size = merged

    def __call__(self, x: mx.array) -> mx.array:
        x = self.ln_q(x).reshape(-1, self.merged_size)
        return self.fc2(nn.gelu(self.fc1(x)))


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embed = nn.Linear(
            config.num_channels * config.patch_size**2, config.hidden_size, bias=False
        )
        self.ln_pre = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.merger = PatchMerger(config)
        self.rope = VisionRotaryEmbedding(config)

    def cu_seqlens(self, grid_thw: mx.array, total_patches: int) -> list[int]:
        """Split each sample into windows of at most frame_windows_size frames."""
        fixed_t = self.config.frame_windows_size
        bounds = [0]
        current = 0
        for row in grid_thw.tolist():
            t, h, w = int(row[0]), int(row[1]), int(row[2])
            spans = []
            if fixed_t and t > fixed_t:
                spans = [fixed_t] * (t // fixed_t)
                if t % fixed_t:
                    spans.append(t % fixed_t)
            else:
                spans = [t]
            for span in spans:
                current += span * h * w
                bounds.append(current)
        if bounds[-1] != total_patches:
            raise ValueError(f"cu_seqlens mismatch: {bounds[-1]} != {total_patches}")
        return bounds

    def __call__(
        self, pixel_values: mx.array, grid_thw: mx.array, patch_positions: mx.array
    ) -> mx.array:
        """pixel_values: [total_patches, C*P*P] -> merged features [total/4, out_hidden]."""
        x = self.patch_embed(pixel_values)[None]
        total_patches = x.shape[1]

        freqs = self.rope.from_positions(patch_positions)
        freqs = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(freqs)[None, None]
        sin = mx.sin(freqs)[None, None]

        x = self.ln_pre(x)
        bounds = self.cu_seqlens(grid_thw, total_patches)
        for layer in self.layers:
            x = layer(x, cos, sin, bounds)
        return self.merger(x)
