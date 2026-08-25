"""Compare this port's prompt building against the official processor.

A wrong visual-slot count shifts every token after it, so the ids must match
exactly for all three input kinds.
"""

import sys
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mage_vl_mlx.codec import preprocess_codec  # noqa: E402
from mage_vl_mlx.prompt import PromptBuilder  # noqa: E402
from mage_vl_mlx.video import preprocess_video  # noqa: E402

MODEL_ID = "microsoft/Mage-VL"
REVISION = "d88b153285f1633a61b2f693c59c8576693af185"
QUESTION = "Describe this media."


def main():
    image_path, video_path = sys.argv[1], sys.argv[2]

    from PIL import Image
    from transformers import AutoProcessor
    from codec_video_processing_mage_vl import CodecConfig, _cache_dir_for  # type: ignore

    checkpoint = Path(snapshot_download(MODEL_ID, revision=REVISION))
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True)
    builder = PromptBuilder(checkpoint)

    def official(kind: str, **kw):
        content = [{"type": kind}, {"type": "text", "text": QUESTION}]
        text = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True)
        return processor(text=[text], return_tensors="pt", **kw)

    results = {}

    ref = official("image", images=[Image.open(image_path).convert("RGB")])
    mine = builder.for_image(QUESTION, ref["image_grid_thw"].numpy())
    results["image"] = (ref["input_ids"].numpy(), mine)

    ref = official("video", videos=[str(Path(video_path).resolve())], padding=True)
    processed = preprocess_video(str(video_path))
    mine = builder.for_video_frames(
        QUESTION, processed["grid_thw"], processed["frame_timestamps"])
    results["video (frames)"] = (ref["input_ids"].numpy(), mine)

    ref = official("video", videos=[str(Path(video_path).resolve())], padding=True,
                   video_backend="codec",
                   codec_config={"patch": 16, "max_pixels": 150000},
                   max_pixels=150000)
    assets = preprocess_codec(
        _cache_dir_for(str(Path(video_path).resolve()),
                       CodecConfig(patch=16, max_pixels=150000)))
    mine = builder.for_video_codec(
        QUESTION, assets["patch_positions"], assets["fps"])
    results["video (codec)"] = (ref["input_ids"].numpy(), mine)

    failed = False
    for name, (reference, actual) in results.items():
        match = reference.shape == actual.shape and bool(np.array_equal(reference, actual))
        failed |= not match
        detail = f"ref={reference.shape[1]} mine={actual.shape[1]}"
        if not match and reference.shape == actual.shape:
            first = int(np.argmax(reference[0] != actual[0]))
            detail += f" first_mismatch_at={first}"
        print(f"{name:16s} match={match}  {detail}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
