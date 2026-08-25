"""Check the MLX streaming gate against Stage 3 fixtures."""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mage_vl_mlx.streaming import StreamMindGate  # noqa: E402


def stats(reference: np.ndarray, actual: np.ndarray) -> dict:
    ref, act = reference.astype(np.float64).ravel(), actual.astype(np.float64).ravel()
    denom = np.linalg.norm(ref)
    return {
        "max_abs_diff": float(np.max(np.abs(ref - act))),
        "rel_error": float(np.linalg.norm(ref - act) / denom),
        "cosine": float(np.dot(ref, act) / (denom * np.linalg.norm(act))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("weights/mage-vl-bf16"))
    parser.add_argument("--fixtures", type=Path, default=Path("fixtures/stage3"))
    parser.add_argument("--dtype", default="float32", choices=("bfloat16", "float32"))
    args = parser.parse_args()

    compute_dtype = getattr(mx, args.dtype)
    gate = StreamMindGate()
    gate.load_weights(str(args.weights / "streammind_gate.safetensors"))
    if compute_dtype != mx.bfloat16:
        gate.update(gate.apply(lambda p: p.astype(compute_dtype)))
    mx.eval(gate.parameters())

    tag = f"cpu-{args.dtype}"
    cases = sorted(p.parent.name for p in args.fixtures.glob(f"*/{tag}.npz"))
    if not cases:
        raise SystemExit(f"no fixtures tagged {tag} under {args.fixtures}")

    report = {}
    for case in cases:
        fixture = np.load(args.fixtures / case / f"{tag}.npz")
        vision_tokens = mx.array(fixture["vision_tokens"]).astype(compute_dtype)

        # The gate threshold is defined on the mixer, so feed it the reference's
        # own input; running end to end would include upstream kernel drift.
        mixer_input = mx.array(fixture["mixer_input"]).astype(compute_dtype)
        mixer_output = gate.mamba_model.ssms[0].mixer(mixer_input)

        pooled = nn.leaky_relu(gate.pre_net.fc3(mx.mean(vision_tokens, axis=2)))
        mamba_out = gate.mamba_model(pooled)
        perception = gate.perception_tokens(vision_tokens)
        logits = gate(vision_tokens)
        mx.eval(mixer_output, mamba_out, perception, logits)

        mixer_stats = stats(
            fixture["mixer_output"], np.array(mixer_output.astype(mx.float32))
        )
        mamba_stats = stats(fixture["mamba_out"], np.array(mamba_out.astype(mx.float32)))
        perception_stats = stats(
            fixture["perception_tokens"], np.array(perception.astype(mx.float32))
        )
        logits_stats = stats(fixture["logits"], np.array(logits.astype(mx.float32)))

        reference_speak = softmax_speak(fixture["logits"])
        actual_speak = softmax_speak(np.array(logits.astype(mx.float32)))
        timeline_match = bool(
            np.array_equal(reference_speak >= 0.5, actual_speak >= 0.5)
        )

        report[case] = {
            "mixer_isolated": mixer_stats,
            "mamba_end_to_end": mamba_stats,
            "perception_tokens": perception_stats,
            "logits": logits_stats,
            "timeline_match": timeline_match,
            "p_speak_max_abs_diff": float(np.max(np.abs(reference_speak - actual_speak))),
            "reference_p_speak": [round(float(v), 6) for v in reference_speak],
        }
        print(f"{case}:")
        print(f"  mixer       max_abs={mixer_stats['max_abs_diff']:.3e} "
              f"rel={mixer_stats['rel_error']:.3e}  (gate: <= 1.0e-5)")
        print(f"  mamba e2e   max_abs={mamba_stats['max_abs_diff']:.3e} "
              f"rel={mamba_stats['rel_error']:.3e}")
        print(f"  perception  max_abs={perception_stats['max_abs_diff']:.3e}")
        print(f"  logits      max_abs={logits_stats['max_abs_diff']:.3e} "
              f"cos={logits_stats['cosine']:.6f}")
        print(f"  timeline    match={timeline_match} "
              f"p_speak_max_diff={report[case]['p_speak_max_abs_diff']:.3e}")
        print(f"  p_speak     {[round(float(v), 4) for v in reference_speak]}")

    out = args.fixtures / f"parity-{tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


def softmax_speak(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted.astype(np.float64))
    return (exp / exp.sum(axis=-1, keepdims=True))[0, :, 1]


if __name__ == "__main__":
    main()
