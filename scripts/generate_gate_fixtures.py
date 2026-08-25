"""Generate Stage 3 streaming-gate fixtures.

Vision tokens come from the official model's own _streammind_vision_tokens.
The gate itself runs through scripts/reference_gate.py, a pure-PyTorch
reimplementation of the SSM block, because mamba-ssm cannot be installed on
macOS. See that module for what this does and does not verify.
"""

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reference_gate import StreamMindGate  # noqa: E402

MODEL_ID = "microsoft/Mage-VL"
REVISION = "d88b153285f1633a61b2f693c59c8576693af185"


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", action="append", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("fixtures/stage3"))
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True,
        dtype=torch.float32, device_map="cpu",
    ).eval()

    gate = StreamMindGate().eval()
    src = Path(snapshot_download(MODEL_ID, revision=REVISION))
    weights = {k: v.float() for k, v in
               load_file(str(src / "streammind_gate.safetensors")).items()}
    missing, unexpected = gate.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise SystemExit(f"gate weights mismatch: {missing} {unexpected}")

    captured: dict = {}
    gate.mamba_model.register_forward_hook(
        lambda _m, _i, out: captured.__setitem__("mamba", to_numpy(out))
    )
    # The Stage 3 gate is defined on the mixer itself, so capture its input and
    # output separately. Comparing end to end would fold in differences the
    # pooling and PreNet introduce before the mixer ever runs.
    block = gate.mamba_model.ssms[0]
    block.norm.register_forward_hook(
        lambda _m, _i, out: captured.__setitem__("mixer_input", to_numpy(out))
    )
    block.mixer.register_forward_hook(
        lambda _m, _i, out: captured.__setitem__("mixer_output", to_numpy(out))
    )

    summary = {
        "model": MODEL_ID, "revision": REVISION, "dtype": "float32", "device": "cpu",
        "gate_reference": "scripts/reference_gate.py (pure-PyTorch SSM)",
        "torch": torch.__version__, "platform": platform.platform(), "cases": {},
    }

    # The gate only consumes the visual tensors, but the processor still needs
    # a prompt carrying the video placeholder.
    text = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "video"}, {"type": "text", "text": "Describe this video."},
        ]}],
        tokenize=False, add_generation_prompt=True,
    )

    for video in args.video:
        inputs = processor(text=[text], videos=[str(video)], return_tensors="pt")
        with torch.inference_mode():
            vision_tokens = model.model._streammind_vision_tokens(
                inputs["pixel_values"].to(model.dtype),
                inputs["image_grid_thw"],
                patch_positions=inputs["patch_positions"],
            )
            perception = gate.perception_tokens(vision_tokens)
            logits = gate(vision_tokens)
            probabilities = torch.softmax(logits.float(), dim=-1)

        speak = probabilities[0, :, 1]
        case_dir = args.out / video.stem
        case_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            case_dir / "cpu-float32.npz",
            pixel_values=to_numpy(inputs["pixel_values"]),
            image_grid_thw=to_numpy(inputs["image_grid_thw"]),
            patch_positions=to_numpy(inputs["patch_positions"]),
            vision_tokens=to_numpy(vision_tokens),
            mamba_out=captured["mamba"],
            mixer_input=captured["mixer_input"],
            mixer_output=captured["mixer_output"],
            perception_tokens=to_numpy(perception),
            logits=to_numpy(logits),
        )
        summary["cases"][video.stem] = {
            "vision_tokens_shape": list(vision_tokens.shape),
            "p_speak": [round(float(v), 6) for v in speak],
            "timeline": ["speak" if float(v) >= 0.5 else "silent" for v in speak],
        }
        print(f"{video.stem}: tokens={list(vision_tokens.shape)}")
        print(f"  p_speak: {[round(float(v), 4) for v in speak]}")

    out_file = args.out / "summary-cpu-float32.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
