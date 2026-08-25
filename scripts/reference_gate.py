"""Pure-PyTorch reference for Mage-VL's StreamMind gate.

The checkpoint's streammind_gate.py builds its SSM with
``mamba_ssm.models.mixer_seq_simple.create_block``, and mamba-ssm cannot be
installed on macOS: its setup.py parses ``torch.version.cuda``, which is None
there. This module reimplements the same block in plain PyTorch, following
mamba_ssm's own published reference semantics (``selective_scan_ref`` and the
non-fast path of ``mamba_simple.Mamba.forward``), so a reference can be
produced on this machine.

Everything outside the SSM — PreNet, PostNet, LayerNorms, and the 4-layer
Qwen3 classifier — uses stock PyTorch and transformers, so those parts are the
official code paths. The SSM itself is compared against this reimplementation
rather than against mamba-ssm's CUDA kernels, which remains unverified.

Weight shapes from streammind_gate.safetensors confirm Mamba1 defaults:
d_model 2560, expand 2 (d_inner 5120), d_state 16, d_conv 4, dt_rank 160,
in_proj/out_proj/x_proj without bias, and a LayerNorm (not RMSNorm) block norm.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3Config
from transformers.models.qwen3 import Qwen3ForCausalLM

D_STATE = 16
D_CONV = 4
EXPAND = 2


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                       delta_softplus=False):
    """mamba_ssm's reference selective scan, variable B/C case."""
    dtype_in = u.dtype
    u, delta = u.float(), delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim = u.shape[0], A.shape[0]
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B.float(), u)

    state = A.new_zeros((batch, dim, A.shape[1]))
    outputs = []
    for i in range(u.shape[2]):
        state = deltaA[:, :, i] * state + deltaB_u[:, :, i]
        outputs.append(torch.einsum("bdn,bn->bd", state, C[:, :, i].float()))
    y = torch.stack(outputs, dim=2)

    if D is not None:
        y = y + u * D.float()[None, :, None]
    if z is not None:
        y = y * F.silu(z.float())
    return y.to(dtype_in)


class MambaMixer(nn.Module):
    """Mamba1 mixer matching mamba_simple.Mamba's non-fast path."""

    def __init__(self, d_model: int = 2560):
        super().__init__()
        self.d_model = d_model
        self.d_inner = EXPAND * d_model
        self.d_state = D_STATE
        self.dt_rank = -(-d_model // 16)  # ceil(d_model / 16), mamba's "auto"

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=D_CONV,
            groups=self.d_inner, padding=D_CONV - 1, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.A_log = nn.Parameter(torch.zeros(self.d_inner, self.d_state))
        self.D = nn.Parameter(torch.zeros(self.d_inner))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states).transpose(1, 2)  # [B, 2*d_inner, L]
        x, z = xz.chunk(2, dim=1)

        x = F.silu(self.conv1d(x)[..., :seqlen])

        x_dbl = self.x_proj(x.transpose(1, 2))  # [B, L, dt_rank + 2*d_state]
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.linear(dt, self.dt_proj.weight).transpose(1, 2)  # bias applied in scan
        B = B.transpose(1, 2).contiguous()
        C = C.transpose(1, 2).contiguous()

        A = -torch.exp(self.A_log.float())
        y = selective_scan_ref(
            x, dt, A, B, C, D=self.D, z=z,
            delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
        )
        return self.out_proj(y.transpose(1, 2))


class MambaBlock(nn.Module):
    """mamba_ssm Block with fused_add_norm=False, rms_norm=False, mlp=Identity."""

    def __init__(self, d_model: int = 2560, norm_epsilon: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=norm_epsilon)
        self.mixer = MambaMixer(d_model)

    def forward(self, hidden_states, residual=None):
        residual = hidden_states if residual is None else hidden_states + residual
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        return self.mixer(hidden_states), residual


class VideoMamba(nn.Module):
    def __init__(self, d_model: int = 2560, n_ssm: int = 1):
        super().__init__()
        self.ssms = nn.ModuleList([MambaBlock(d_model) for _ in range(n_ssm)])
        self.norm_fn = nn.LayerNorm(d_model)

    def forward(self, embeds):
        hidden_states, residual = embeds, None
        for ssm in self.ssms:
            hidden_states, residual = ssm(hidden_states, residual)
        residual = hidden_states + residual if residual is not None else hidden_states
        return self.norm_fn(residual.to(dtype=self.norm_fn.weight.dtype))


class PreNet(nn.Module):
    def __init__(self, d_code, d_model):
        super().__init__()
        self.fc3 = nn.Linear(d_code, d_model)

    def forward(self, x):
        return F.leaky_relu(self.fc3(x))


class PostNet(nn.Module):
    def __init__(self, d_model, n_class):
        super().__init__()
        self.fc3 = nn.Linear(d_model, n_class)

    def forward(self, x):
        return self.fc3(F.leaky_relu(x))


class Qwen3ForCausalLMCls(Qwen3ForCausalLM):
    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return self.lm_head(outputs.last_hidden_state).float()


class ClsNet(nn.Module):
    def __init__(self, hidden_size=2560, num_layers=4):
        super().__init__()
        config = Qwen3Config(
            vocab_size=2, hidden_size=hidden_size, num_hidden_layers=num_layers,
            num_attention_heads=32, num_key_value_heads=8, intermediate_size=12288,
            head_dim=128, max_position_embeddings=8192, rms_norm_eps=1e-6,
            tie_word_embeddings=False, attention_bias=False,
        )
        self.cls_model = Qwen3ForCausalLMCls(config)

    def forward(self, x, attention_mask=None):
        return self.cls_model(inputs_embeds=x, attention_mask=attention_mask)


class StreamMindGate(nn.Module):
    def __init__(self, hidden_size: int = 2560):
        super().__init__()
        self.pre_net = PreNet(hidden_size, hidden_size)
        self.mamba_model = VideoMamba(d_model=hidden_size)
        self.post_net = PostNet(hidden_size, hidden_size)
        self.cls_net = ClsNet(hidden_size=hidden_size, num_layers=4)

    def perception_tokens(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        """[B, T, P, D] visual patches -> one EPFE token per time step."""
        x = vision_tokens.mean(dim=2)
        batch, time, dim = x.shape
        x = self.pre_net(x.reshape(batch * time, dim)).reshape(batch, time, dim)
        x = self.mamba_model(x)
        return self.post_net(x.reshape(batch * time, dim)).reshape(batch, time, dim)

    def forward(self, vision_tokens, response_positions=None):
        """Return [B, T, 2] silent/speak logits for each EPFE time step."""
        tokens = self.perception_tokens(vision_tokens)
        batch, time, dim = tokens.shape
        target_ids = torch.zeros(batch, time, dtype=torch.long, device=tokens.device)
        if response_positions is not None:
            positions = torch.as_tensor(response_positions, device=tokens.device) - 1
            target_ids[:, positions] = 1
        targets = self.cls_net.cls_model.model.embed_tokens(target_ids.reshape(batch * time))
        pair = torch.stack((tokens.reshape(batch * time, dim), targets), dim=1)

        rotary = self.cls_net.cls_model.model.rotary_emb
        saved_inv_freq = rotary.inv_freq
        try:
            rotary.inv_freq = rotary.inv_freq.to(pair.dtype)
            logits = self.cls_net(
                pair, attention_mask=torch.ones(pair.shape[:2], device=pair.device)
            )
        finally:
            rotary.inv_freq = saved_inv_freq
        return logits[:, 0].reshape(batch, time, 2)
