"""Convert the official Mage-VL checkpoint to this port's MLX layout.

Reports missing and unused weight keys, which is the first Stage 1 gate.
"""

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mage_vl_mlx.config import MageVLConfig  # noqa: E402
from mage_vl_mlx.model import MageVL  # noqa: E402

MODEL_ID = "microsoft/Mage-VL"
REVISION = "d88b153285f1633a61b2f693c59c8576693af185"


def rename(key: str) -> str | None:
    if key == "lm_head.weight":
        return key
    if key.startswith("model.language_model."):
        return "language." + key[len("model.language_model."):]
    if key.startswith("model.visual."):
        rest = key[len("model.visual."):]
        if rest.startswith("embeddings.patch_embedding."):
            return "vision.patch_embed." + rest.split(".")[-1]
        if rest.startswith("layernorm_pre."):
            return "vision.ln_pre." + rest.split(".")[-1]
        if rest.startswith("encoder.layers."):
            return "vision.layers." + rest[len("encoder.layers."):]
        if rest.startswith("merger.ln_q."):
            return "vision.merger.ln_q." + rest.split(".")[-1]
        if rest.startswith("merger.mlp."):
            parts = rest.split(".")  # merger, mlp, <idx>, <param>
            fc = {"0": "fc1", "2": "fc2"}.get(parts[2])
            if fc is None:
                return None
            return f"vision.merger.{fc}.{parts[3]}"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("weights/mage-vl-bf16"))
    args = parser.parse_args()

    src = Path(snapshot_download(MODEL_ID, revision=REVISION))
    index = json.loads((src / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))

    converted: dict[str, mx.array] = {}
    skipped: list[str] = []
    for shard in shards:
        for key, array in mx.load(str(src / shard)).items():
            new_key = rename(key)
            if new_key is None:
                skipped.append(key)
                continue
            if new_key == "vision.patch_embed.weight":
                # Conv2d(kernel=stride=patch_size) over a single patch is a matmul.
                array = array.reshape(array.shape[0], -1)
            converted[new_key] = array.astype(mx.bfloat16)

    config = MageVLConfig.from_json(src / "config.json")
    model = MageVL(config)
    expected = {k for k, _ in tree_flatten(model.parameters())}

    missing = sorted(expected - converted.keys())
    unused = sorted(converted.keys() - expected)
    print(f"converted={len(converted)} expected={len(expected)}")
    print(f"missing={len(missing)} unused={len(unused)} skipped_source_keys={len(skipped)}")
    for key in missing[:10]:
        print("  missing:", key)
    for key in unused[:10]:
        print("  unused:", key)
    for key in skipped[:10]:
        print("  skipped:", key)
    if missing or unused:
        raise SystemExit("weight key mismatch")

    shapes_ok = True
    params = dict(tree_flatten(model.parameters()))
    for key, value in converted.items():
        if params[key].shape != value.shape:
            print(f"  shape mismatch {key}: {params[key].shape} != {value.shape}")
            shapes_ok = False
    if not shapes_ok:
        raise SystemExit("weight shape mismatch")

    args.out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.out / "model.safetensors"), converted)
    shutil.copy(src / "config.json", args.out / "config.json")
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "preprocessor_config.json", "vocab.json", "merges.txt",
                 "added_tokens.json", "special_tokens_map.json"):
        if (src / name).exists():
            shutil.copy(src / name, args.out / name)
    print(f"wrote {args.out / 'model.safetensors'}")


if __name__ == "__main__":
    main()
