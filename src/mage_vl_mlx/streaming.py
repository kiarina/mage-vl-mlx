"""StreamMind proactive streaming gate in MLX.

The gate mean-pools each frame's visual patches into one EPFE token, runs them
through a Mamba1 SSM, and classifies every time step as silent or speak with a
4-layer Qwen3 head. Note the head uses rope_theta 10000, not the main decoder's
5e6 — it is built from Qwen3Config defaults.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .config import TextConfig
from .language import LanguageModel

D_STATE = 16
D_CONV = 4
EXPAND = 2


def cls_config(hidden_size: int = 2560, num_layers: int = 4) -> TextConfig:
    return TextConfig(
        hidden_size=hidden_size, intermediate_size=12288,
        num_hidden_layers=num_layers, num_attention_heads=32,
        num_key_value_heads=8, head_dim=128, rms_norm_eps=1e-6,
        vocab_size=2, rope_theta=10000.0, tie_word_embeddings=False,
    )


class MambaMixer(nn.Module):
    """Mamba1 mixer following mamba_ssm's reference (non-fast) path."""

    def __init__(self, d_model: int = 2560):
        super().__init__()
        self.d_model = d_model
        self.d_inner = EXPAND * d_model
        self.d_state = D_STATE
        self.dt_rank = -(-d_model // 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=D_CONV,
            padding=D_CONV - 1, groups=self.d_inner, bias=True,
        )
        self.A_log = mx.zeros((self.d_inner, self.d_state))
        self.D = mx.zeros((self.d_inner,))

    def __call__(self, hidden_states: mx.array) -> mx.array:
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        x, z = mx.split(xz, [self.d_inner], axis=-1)  # [B, L, d_inner] each

        # mlx conv1d takes [N, L, C]; truncate the causal padding tail.
        x = nn.silu(self.conv1d(x)[:, :seqlen])

        projected = self.x_proj(x)
        dt, b_mat, c_mat = mx.split(
            projected, [self.dt_rank, self.dt_rank + self.d_state], axis=-1
        )
        # dt_proj's bias is applied inside the scan, before softplus.
        dt = dt @ self.dt_proj.weight.T

        # The scan runs in float32 regardless of the weights' dtype, matching
        # selective_scan_ref, which upcasts before accumulating.
        dt = nn.softplus(dt.astype(mx.float32) + self.dt_proj.bias.astype(mx.float32))
        a_mat = -mx.exp(self.A_log.astype(mx.float32))
        x32 = x.astype(mx.float32)
        b32, c32 = b_mat.astype(mx.float32), c_mat.astype(mx.float32)

        delta_a = mx.exp(dt[..., None] * a_mat)             # [B, L, d_inner, d_state]
        delta_b_u = dt[..., None] * b32[:, :, None, :] * x32[..., None]

        state = mx.zeros(delta_a.shape[:1] + delta_a.shape[2:], dtype=mx.float32)
        outputs = []
        for i in range(seqlen):
            state = delta_a[:, i] * state + delta_b_u[:, i]
            outputs.append(mx.sum(state * c32[:, i][:, None, :], axis=-1))
        y = mx.stack(outputs, axis=1)                        # [B, L, d_inner]

        y = y + x32 * self.D.astype(mx.float32)
        y = y * nn.silu(z.astype(mx.float32))
        return self.out_proj(y.astype(hidden_states.dtype))


class MambaBlock(nn.Module):
    def __init__(self, d_model: int = 2560, norm_eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=norm_eps)
        self.mixer = MambaMixer(d_model)

    def __call__(self, hidden_states, residual):
        residual = hidden_states if residual is None else hidden_states + residual
        return self.mixer(self.norm(residual)), residual


class VideoMamba(nn.Module):
    def __init__(self, d_model: int = 2560, n_ssm: int = 1):
        super().__init__()
        self.ssms = [MambaBlock(d_model) for _ in range(n_ssm)]
        self.norm_fn = nn.LayerNorm(d_model)

    def __call__(self, embeds: mx.array) -> mx.array:
        hidden_states, residual = embeds, None
        for ssm in self.ssms:
            hidden_states, residual = ssm(hidden_states, residual)
        return self.norm_fn(hidden_states + residual)


class FeedForward(nn.Module):
    """PreNet and PostNet share one Linear named fc3 in the checkpoint."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.fc3 = nn.Linear(d_in, d_out, bias=True)


class ClsHead(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.model = LanguageModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, embeds: mx.array) -> mx.array:
        return self.lm_head(self.model(embeds))


class StreamMindGate(nn.Module):
    def __init__(self, hidden_size: int = 2560):
        super().__init__()
        self.pre_net = FeedForward(hidden_size, hidden_size)
        self.mamba_model = VideoMamba(d_model=hidden_size)
        self.post_net = FeedForward(hidden_size, hidden_size)
        self.cls_net = ClsHead(cls_config(hidden_size))

    def perception_tokens(self, vision_tokens: mx.array) -> mx.array:
        """[B, T, P, D] visual patches -> one EPFE token per time step."""
        x = mx.mean(vision_tokens, axis=2)
        x = nn.leaky_relu(self.pre_net.fc3(x))
        x = self.mamba_model(x)
        return self.post_net.fc3(nn.leaky_relu(x))

    def __call__(
        self, vision_tokens: mx.array, response_positions: list[int] | None = None
    ) -> mx.array:
        """Return [B, T, 2] silent/speak logits for each EPFE time step."""
        tokens = self.perception_tokens(vision_tokens)
        batch, time, dim = tokens.shape

        target_ids = mx.zeros((batch, time), dtype=mx.int32)
        if response_positions:
            index = mx.array([p - 1 for p in response_positions], dtype=mx.int32)
            target_ids[:, index] = mx.ones((batch, index.size), dtype=mx.int32)
        targets = self.cls_net.model.embed_tokens(target_ids.reshape(batch * time))

        # Each time step is classified from a length-2 sequence: the EPFE token
        # followed by the target embedding; position 0 predicts position 1.
        pair = mx.stack([tokens.reshape(batch * time, dim), targets], axis=1)
        logits = self.cls_net(pair)
        return logits[:, 0].reshape(batch, time, 2)
