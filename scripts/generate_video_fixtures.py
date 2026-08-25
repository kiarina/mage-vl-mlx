"""Generate Stage 2 video fixtures from the official PyTorch implementation.

Passes video paths through MageVLProcessor, the same call official inference
makes. That routes video through the image path: pixel_values with one
image_grid_thw row per frame, and patch_positions carrying real frame indices.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_ID = "microsoft/Mage-VL"
REVISION = "d88b153285f1633a61b2f693c59c8576693af185"
DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", action="append", required=True, type=Path)
    parser.add_argument("--question", default="Describe this video.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=sorted(DTYPES))
    parser.add_argument("--out", type=Path, default=Path("fixtures/stage2"))
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True,
        dtype=DTYPES[args.dtype], device_map=args.device,
    ).eval()

    captured: dict = {}

    def visual_hook(_module, _inputs, output):
        tensor = getattr(output, "last_hidden_state", None)
        if tensor is None:
            tensor = output[0] if isinstance(output, tuple) else output
        captured["visual"] = to_numpy(tensor)

    model.model.visual.register_forward_hook(visual_hook)

    messages = [{"role": "user", "content": [
        {"type": "video"}, {"type": "text", "text": args.question},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    summary = {
        "model": MODEL_ID, "revision": REVISION, "question": args.question,
        "max_new_tokens": args.max_new_tokens, "device": args.device,
        "dtype": args.dtype, "torch": torch.__version__,
        "platform": platform.platform(), "cases": {},
    }

    for video in args.video:
        inputs = processor(text=[text], videos=[str(video)], return_tensors="pt")
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        for key in ("pixel_values_videos", "pixel_values"):
            if key in inputs:
                inputs[key] = inputs[key].to(model.dtype)

        start = time.perf_counter()
        with torch.inference_mode():
            forward_out = model(**inputs)
            output = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
            )
        wall_s = time.perf_counter() - start

        prompt_len = inputs["input_ids"].shape[1]
        greedy_ids = output[0, prompt_len:]
        answer = processor.tokenizer.decode(greedy_ids, skip_special_tokens=True)

        case_dir = args.out / video.stem
        case_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            case_dir / f"{args.device}-{args.dtype}.npz",
            input_ids=to_numpy(inputs["input_ids"]),
            pixel_values=to_numpy(inputs["pixel_values"]),
            image_grid_thw=to_numpy(inputs["image_grid_thw"]),
            patch_positions=to_numpy(inputs["patch_positions"]),
            visual=captured["visual"],
            logits_last=to_numpy(forward_out.logits[0, -1]),
            greedy_ids=to_numpy(greedy_ids),
        )
        summary["cases"][video.stem] = {
            "image_grid_thw": inputs["image_grid_thw"].tolist(),
            "prompt_tokens": int(prompt_len),
            "visual_shape": list(captured["visual"].shape),
            "wall_s": round(wall_s, 2),
            "answer": answer.strip(),
        }
        print(f"[{args.device}/{args.dtype}] {video.stem}: "
              f"grid={inputs["image_grid_thw"].shape[0]}x{inputs["image_grid_thw"][0].tolist()} prompt={prompt_len} "
              f"wall={wall_s:.1f}s")
        print(f"  answer: {answer.strip()[:110]}")

    out_file = args.out / f"summary-{args.device}-{args.dtype}.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
