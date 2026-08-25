"""Convert streammind_gate.safetensors to this port's MLX layout."""

import argparse
import sys
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mage_vl_mlx.streaming import StreamMindGate  # noqa: E402

MODEL_ID = "microsoft/Mage-VL"
REVISION = "d88b153285f1633a61b2f693c59c8576693af185"
CLS_PREFIX = "cls_net.cls_model."


def rename(key: str) -> str:
    if key.startswith(CLS_PREFIX):
        rest = key[len(CLS_PREFIX):]
        if rest.startswith("model."):
            return "cls_net.model." + rest[len("model."):]
        return "cls_net." + rest
    return key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("weights/mage-vl-bf16"))
    args = parser.parse_args()

    src = Path(snapshot_download(MODEL_ID, revision=REVISION))
    raw = mx.load(str(src / "streammind_gate.safetensors"))

    converted = {}
    for key, array in raw.items():
        new_key = rename(key)
        if new_key.endswith("mixer.conv1d.weight"):
            # torch [C, 1, K] -> mlx [C, K, 1]
            array = array.transpose(0, 2, 1)
        converted[new_key] = array.astype(mx.bfloat16)

    model = StreamMindGate()
    expected = {k for k, _ in tree_flatten(model.parameters())}
    missing = sorted(expected - converted.keys())
    unused = sorted(converted.keys() - expected)
    print(f"converted={len(converted)} expected={len(expected)}")
    print(f"missing={len(missing)} unused={len(unused)}")
    for key in missing[:10]:
        print("  missing:", key)
    for key in unused[:10]:
        print("  unused:", key)
    if missing or unused:
        raise SystemExit("gate weight key mismatch")

    params = dict(tree_flatten(model.parameters()))
    bad = [k for k, v in converted.items() if params[k].shape != v.shape]
    for key in bad[:10]:
        print(f"  shape mismatch {key}: {params[key].shape} != {converted[key].shape}")
    if bad:
        raise SystemExit("gate weight shape mismatch")

    args.out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.out / "streammind_gate.safetensors"), converted)
    print(f"wrote {args.out / 'streammind_gate.safetensors'}")


if __name__ == "__main__":
    main()
