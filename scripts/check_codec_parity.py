"""End-to-end codec-path parity: official PyTorch (fp32 CPU) vs this MLX port."""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mage_vl_mlx.codec import preprocess_codec  # noqa: E402
from mage_vl_mlx.model import MageVL  # noqa: E402

MODEL_ID = "microsoft/Mage-VL"
REVISION = "d88b153285f1633a61b2f693c59c8576693af185"


def stats(reference, actual):
    ref, act = reference.astype(np.float64).ravel(), actual.astype(np.float64).ravel()
    denom = np.linalg.norm(ref)
    return {"rel_error": float(np.linalg.norm(ref - act) / denom),
            "cosine": float(np.dot(ref, act) / (denom * np.linalg.norm(act)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", action="append", required=True, type=Path)
    parser.add_argument("--weights", type=Path, default=Path("weights/mage-vl-bf16"))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--out", type=Path, default=Path("fixtures/stage4"))
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoProcessor
    from codec_video_processing_mage_vl import CodecConfig, _cache_dir_for  # type: ignore

    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True)
    torch_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True,
        dtype=torch.float32, device_map="cpu").eval()

    model = MageVL.from_pretrained(args.weights)
    model.update(model.apply(lambda p: p.astype(mx.float32)))
    mx.eval(model.parameters())

    text = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "video"}, {"type": "text", "text": "Describe this video."}]}],
        tokenize=False, add_generation_prompt=True)

    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for video in args.video:
        path = str(video.resolve())
        inputs = processor(
            text=[text], videos=[path], video_backend="codec",
            codec_config={"patch": 16, "max_pixels": 150000}, max_pixels=150000,
            return_tensors="pt", padding=True)
        inputs = {k: v for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)

        with torch.inference_mode():
            forward_out = torch_model(**inputs)
            output = torch_model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        reference_ids = output[0, prompt_len:].tolist()
        reference_text = processor.tokenizer.decode(
            output[0, prompt_len:], skip_special_tokens=True).strip()
        reference_logits = forward_out.logits[0, -1].float().numpy()

        processed = preprocess_codec(_cache_dir_for(path, CodecConfig(patch=16, max_pixels=150000)))
        ids = mx.array(inputs["input_ids"].numpy().astype(np.int32))
        pixel_values = mx.array(processed["pixel_values"])
        grid = mx.array(processed["grid_thw"].astype(np.int32))
        positions = mx.array(processed["patch_positions"].astype(np.int32))

        logits = model(model.embed(ids, pixel_values, grid, positions))[:, -1]
        mx.eval(logits)
        generated = model.generate(ids, pixel_values, grid, positions,
                                   max_new_tokens=args.max_new_tokens)
        matched = sum(1 for a, b in zip(generated, reference_ids) if a == b)

        report[video.stem] = {
            "canvases": processed["canvas_count"],
            "prompt_tokens": int(prompt_len),
            "visual_tokens": int(inputs["image_grid_thw"].prod(-1).sum()) // 4,
            "logits_last": stats(reference_logits, np.array(logits[0].astype(mx.float32))),
            "greedy_match": generated == reference_ids,
            "greedy_matched_tokens": f"{matched}/{len(reference_ids)}",
            "reference_text": reference_text,
        }
        entry = report[video.stem]
        print(f"{video.stem}: canvases={entry['canvases']} "
              f"visual_tokens={entry['visual_tokens']} prompt={entry['prompt_tokens']}")
        print(f"  logits rel={entry['logits_last']['rel_error']:.3e} "
              f"cos={entry['logits_last']['cosine']:.6f}")
        print(f"  greedy {entry['greedy_matched_tokens']} match={entry['greedy_match']}")
        print(f"  text: {reference_text[:100]}")

    (args.out / "codec-parity.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out / 'codec-parity.json'}")


if __name__ == "__main__":
    main()
