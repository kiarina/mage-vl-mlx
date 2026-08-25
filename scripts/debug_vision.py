"""Compare the MLX vision tower against PyTorch stage by stage.

Hooks the reference implementation's submodules and reports where the two
diverge, so a systematic error can be isolated from bf16 rounding noise.
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mage_vl_mlx.model import MageVL  # noqa: E402

MODEL_ID = "microsoft/Mage-VL"
REVISION = "d88b153285f1633a61b2f693c59c8576693af185"
FIXTURE = Path("fixtures/stage1/objects_1536x1024_358kb/mps.npz")


def report(label: str, ref: np.ndarray, act: np.ndarray) -> None:
    r, a = ref.astype(np.float64).ravel(), act.astype(np.float64).ravel()
    rel = np.linalg.norm(r - a) / np.linalg.norm(r)
    cos = np.dot(r, a) / (np.linalg.norm(r) * np.linalg.norm(a))
    print(f"{label:28s} rel={rel:.3e} cos={cos:.6f} max_abs={np.max(np.abs(r - a)):.3e}")


def main():
    fixture = np.load(FIXTURE)
    pixel_values = fixture["pixel_values"]
    grid_thw = fixture["image_grid_thw"].astype(np.int64)
    patch_positions = fixture["patch_positions"].astype(np.int64)

    torch_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True,
        dtype=torch.bfloat16, device_map="mps",
    ).eval()
    visual = torch_model.model.visual

    captured: dict[str, np.ndarray] = {}

    def grab(name):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            captured[name] = t.detach().to(torch.float32).cpu().numpy()
        return hook

    visual.embeddings.register_forward_hook(grab("embeddings"))
    visual.layernorm_pre.register_forward_hook(grab("layernorm_pre"))
    for i in (0, 1, 11, 23):
        visual.encoder.layers[i].register_forward_hook(grab(f"layer{i}"))
    visual.encoder.layers[0].self_attn.register_forward_hook(grab("layer0.attn"))
    visual.encoder.layers[0].layer_norm1.register_forward_hook(grab("layer0.ln1"))
    visual.merger.register_forward_hook(grab("merger"))

    with torch.inference_mode():
        visual(
            torch.tensor(pixel_values, dtype=torch.bfloat16, device="mps"),
            grid_thw=torch.tensor(grid_thw, device="mps"),
            patch_positions=torch.tensor(patch_positions, device="mps"),
        )

    model = MageVL.from_pretrained("weights/mage-vl-bf16")
    vision = model.vision
    px = mx.array(pixel_values).astype(mx.bfloat16)
    pos = mx.array(patch_positions.astype(np.int32))
    thw = mx.array(grid_thw.astype(np.int32))

    def np32(x):
        mx.eval(x)
        return np.array(x.astype(mx.float32))

    x = vision.patch_embed(px)[None]
    report("embeddings", captured["embeddings"], np32(x)[0])

    freqs = vision.rope.from_positions(pos)
    freqs = mx.concatenate([freqs, freqs], axis=-1)
    cos_e, sin_e = mx.cos(freqs)[None, None], mx.sin(freqs)[None, None]

    x = vision.ln_pre(x)
    report("layernorm_pre", captured["layernorm_pre"], np32(x)[0])

    layer0 = vision.layers[0]
    h = layer0.layer_norm1(x)
    report("layer0.ln1", captured["layer0.ln1"], np32(h)[0])

    bounds = vision.cu_seqlens(thw, x.shape[1])
    attn = layer0.self_attn(h, cos_e, sin_e, bounds)
    report("layer0.attn", captured["layer0.attn"], np32(attn)[0])

    for i, layer in enumerate(vision.layers):
        x = layer(x, cos_e, sin_e, bounds)
        if i in (0, 1, 11, 23):
            report(f"layer{i}", captured[f"layer{i}"], np32(x)[0])

    merged = vision.merger(x)
    report("merger", captured["merger"], np32(merged))


if __name__ == "__main__":
    main()
