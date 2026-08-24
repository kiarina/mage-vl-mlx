"""Generate parity fixtures from the official Mage-VL PyTorch implementation.

For each image and device, runs the pinned microsoft/Mage-VL checkpoint with
greedy decoding and captures:

- processor outputs (input_ids, pixel_values, image_grid_thw)
- vision tower output (forward hook on model.model.visual)
- logits at the last prompt position
- greedy generated token ids and decoded text

Results are written to <out>/<image-stem>/<device>.npz plus a summary.json.
When two or more devices are given, prints cross-device diffs so the fixture
device policy can be decided from data.
"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_ID = "microsoft/Mage-VL"
REVISION = "d88b153285f1633a61b2f693c59c8576693af185"


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).cpu().numpy()


def run_device(device: str, images: list[Path], question: str,
               max_new_tokens: int, out_dir: Path) -> dict:
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map=device,
    ).eval()

    captured: dict = {}

    def visual_hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        captured["visual"] = to_numpy(tensor)

    model.model.visual.register_forward_hook(visual_hook)

    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    results = {}
    for image_path in images:
        inputs = processor(
            text=[text],
            images=[Image.open(image_path).convert("RGB")],
            return_tensors="pt",
        )
        inputs = {
            k: (v.to(model.device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

        t0 = time.perf_counter()
        with torch.inference_mode():
            forward_out = model(**inputs)
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        wall_s = time.perf_counter() - t0

        prompt_len = inputs["input_ids"].shape[1]
        greedy_ids = output[0, prompt_len:]
        answer = processor.tokenizer.decode(greedy_ids, skip_special_tokens=True)

        stem = image_path.stem
        case_dir = out_dir / stem
        case_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            case_dir / f"{device}.npz",
            input_ids=to_numpy(inputs["input_ids"]),
            pixel_values=to_numpy(inputs["pixel_values"]),
            image_grid_thw=to_numpy(inputs["image_grid_thw"]),
            visual=captured["visual"],
            logits_last=to_numpy(forward_out.logits[0, -1]),
            greedy_ids=to_numpy(greedy_ids),
        )
        results[stem] = {
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "prompt_tokens": int(prompt_len),
            "visual_shape": list(captured["visual"].shape),
            "greedy_tokens": int(greedy_ids.shape[0]),
            "wall_s": round(wall_s, 2),
            "answer": answer.strip(),
        }
        print(f"[{device}] {stem}: visual{results[stem]['visual_shape']} "
              f"prompt={prompt_len} wall={wall_s:.1f}s")
    return results


def compare(out_dir: Path, devices: list[str], cases: list[str]) -> dict:
    diffs: dict = {}
    base, other = devices[0], devices[1]
    for stem in cases:
        a = np.load(out_dir / stem / f"{base}.npz")
        b = np.load(out_dir / stem / f"{other}.npz")
        entry = {}
        for key in ("pixel_values", "visual", "logits_last"):
            x, y = a[key].ravel(), b[key].ravel()
            cos = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
            entry[key] = {
                "max_abs_diff": float(np.max(np.abs(x - y))),
                "cosine": cos,
            }
        entry["greedy_match"] = bool(np.array_equal(a["greedy_ids"], b["greedy_ids"]))
        entry["greedy_first_mismatch"] = (
            None if entry["greedy_match"]
            else int(np.argmax(a["greedy_ids"] != b["greedy_ids"]))
        )
        diffs[stem] = entry
        print(f"[compare {base} vs {other}] {stem}: "
              f"visual max_abs={entry['visual']['max_abs_diff']:.3e} "
              f"cos={entry['visual']['cosine']:.6f} "
              f"logits max_abs={entry['logits_last']['max_abs_diff']:.3e} "
              f"greedy_match={entry['greedy_match']}")
    return diffs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", required=True, type=Path)
    parser.add_argument("--question", default="Describe this image.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--devices", default="cpu,mps")
    parser.add_argument("--out", type=Path, default=Path("fixtures/stage1"))
    args = parser.parse_args()

    devices = args.devices.split(",")
    args.out.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": MODEL_ID,
        "revision": REVISION,
        "question": args.question,
        "max_new_tokens": args.max_new_tokens,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "devices": {},
    }
    for device in devices:
        summary["devices"][device] = run_device(
            device, args.image, args.question, args.max_new_tokens, args.out
        )
    if len(devices) >= 2:
        summary["diffs"] = compare(
            args.out, devices, [p.stem for p in args.image]
        )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
