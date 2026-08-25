"""Qwen3 decoder in MLX, as used by Mage-VL."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .config import TextConfig


class KVCache:
    """Growing key/value cache for one attention layer."""

    def __init__(self):
        self.keys: mx.array | None = None
        self.values: mx.array | None = None

    @property
    def offset(self) -> int:
        return 0 if self.keys is None else self.keys.shape[2]

    def update(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        if self.keys is None:
            self.keys, self.values = keys, values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        return self.keys, self.values


class Attention(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rope_theta = config.rope_theta

    def __call__(self, x: mx.array, mask, cache: KVCache | None) -> mx.array:
        b, seq, _ = x.shape
        q = self.q_proj(x).reshape(b, seq, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, seq, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, seq, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.q_norm(q)
        k = self.k_norm(k)

        offset = cache.offset if cache is not None else 0
        q = mx.fast.rope(q, self.head_dim, traditional=False, base=self.rope_theta,
                         scale=1.0, offset=offset)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=self.rope_theta,
                         scale=1.0, offset=offset)

        if cache is not None:
            k, v = cache.update(k, v)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(b, seq, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, mask, cache: KVCache | None) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, embeds: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        mask = None
        if embeds.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(embeds.shape[1])
            mask = mask.astype(embeds.dtype)

        x = embeds
        for i, layer in enumerate(self.layers):
            x = layer(x, mask, None if cache is None else cache[i])
        return self.norm(x)
