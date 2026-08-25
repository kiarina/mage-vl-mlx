# mage-vl-mlx

An independent MLX port of [Microsoft Mage-VL](https://huggingface.co/microsoft/Mage-VL)
(4B codec-native vision-language model) for Apple Silicon, built without
depending on mlx-vlm.

The goal is to reproduce all Mage-VL paths — static images, frame-sampled
video, proactive streaming, and the codec-native sparse video path — with
quantitative parity against the official PyTorch implementation.

## Status

All four stages pass in float32. Images, frame-sampled video, the proactive
streaming gate, and the codec-native sparse video path work end to end.

The staged verification plan, gates, and per-stage lab records live in
[kiarina/labs](https://github.com/kiarina/labs/blob/main/docs/mage-vl-mlx-port.md).

| Stage | Scope | Status |
|---|---|---|
| 0 | Codec preprocessing portability | done (conditional pass) |
| 1 | Static image parity | passed in float32 |
| 2 | Torch-free frame-sampled video | passed in float32 |
| 3 | Proactive streaming gate | passed in float32 (see caveat) |
| 4 | Codec-native sparse video | passed in float32 |

## Usage

Two scripts mirror the CLI of microsoft/Mage's `mage_vl/`. Online mode is
deliberately absent: it exists to talk to a CUDA SGLang server, which is the
opposite of what this port is for.

```sh
uv sync
python scripts/convert_weights.py        # -> weights/mage-vl-bf16
python scripts/convert_gate_weights.py   # streaming gate

# image
python inference_base.py --mode offline --image photo.jpg \
  --question "Describe this image."

# frame-sampled video
python inference_base.py --mode offline --video clip.mp4 \
  --video-backend frames

# codec-native video (needs the container wrapper below)
export CV_PREINFER_BIN=$PWD/docker/cv-preinfer
python inference_base.py --mode offline --video clip.mp4 --video-backend codec

# event-gated streaming
python inference_streaming.py --video clip.mp4 --segment-sec 4
```

`--verbose` on `inference_base.py` prints load time, prompt length, tokens/s,
and peak memory to stderr.

The codec backend needs `cv-preinfer`, which ships Linux-only wheels. Build the
container once and the scripts drive it transparently:

```sh
docker build --platform linux/arm64 -t mage-cvprep:0.2.5 -f docker/Dockerfile.cvprep docker/
```

`inference_streaming.py` runs in float32 by default, because the gate's
decision flips under bfloat16 when a score sits near the threshold. Note what
the gate actually decides: it separates content types — a sports broadcast
scores ~0.7-0.8, a quiet hallway ~0.05-0.11 — but it does **not** track event
times within a stream. It answers "is this stream worth commentating on", not
"did something just happen".

Prompt building (`mage_vl_mlx.prompt`) uses `tokenizers` and `jinja2`, neither
of which pulls in torch. Its token ids match the official processor exactly for
images, frame-sampled video, and codec video.

### Stage 1 results

Weight keys: 696 mapped, 0 missing, 0 unused, 0 shape mismatches.

Against float32 CPU fixtures (3 images, greedy 64 tokens):

| Image | vision rel err | vision cosine | greedy |
|---|---:|---:|---:|
| objects | 1.56e-05 | 1.000000 | 64/64 |
| ocr | 1.15e-05 | 1.000000 | 64/64 |
| street_scene | 8.93e-06 | 1.000000 | 64/64 |

Against bfloat16 MPS fixtures the same code diverges (vision cosine
0.9988–0.9992, greedy 7–61 of 64). The logic is identical, so this is
accumulated bfloat16 rounding differing between MLX and PyTorch-MPS
kernels — PyTorch's own CPU/MPS bf16 gap on this model is cosine 0.99953.
Compare in float32; treat bfloat16 as the deployment precision.

Speed on an M4 Max (bfloat16, 1561-token prompt, greedy 64, 3 runs):
21.9 tokens/s, MLX peak memory 9.88 GB.

### Stage 2 results

Preprocessing (`mage_vl_mlx.video`) uses only OpenCV, NumPy, and PIL. On
three 8-frame clips — including one whose frame size forces a resize — the
selected frame indices, grid, patch positions, and pixel values are
**bit-identical** to the official processor, and greedy 64-token output
matches on all three in float32.

| Video | preprocess | vision rel err | greedy |
|---|---|---:|---:|
| pan_objects | bit-exact | 3.61e-04 | 64/64 |
| street_ocr | bit-exact | 1.82e-05 | 64/64 |
| faces_odd (resized) | bit-exact | 8.89e-06 | 64/64 |

Two details a port has to get right, both verified here:

- `patch_positions`'s t axis carries **real source frame indices**
  (e.g. 0, 17, 34, …, 119), not a dense 0..T-1 range.
- `MageVLProcessor` routes video through the image path and emits **one
  `image_grid_thw` row per frame**, so the vision tower attends within each
  frame. The standalone `MageVLVideoProcessor` instead returns a merged
  `[T, h, w]` row, which with `frame_windows_size=4` would attend across
  four frames. Pixel values and patch positions are identical either way;
  only the attention windows differ. This port follows the inference path.

Video speed on an M4 Max (bfloat16, 8 frames, 3159-token prompt, greedy 64):
14.4 tokens/s, MLX peak memory 10.66 GB, preprocessing 0.15 s.

### Stage 3 results

The gate (`mage_vl_mlx.streaming`) mean-pools each frame's patches into one
EPFE token, runs a Mamba1 SSM, and classifies every step silent/speak with a
4-layer Qwen3 head — note that head uses rope_theta 10000, not the decoder's
5e6. All 64 gate weights map with no missing or unused keys.

Against float32 fixtures on four 8-frame clips, the mixer — fed the
reference's own input, which is what the gate threshold is about — matches to
**2.7e-07..4.4e-07 max abs**, against a 1.0e-5 bar, and the speak/silent
timeline matches on all four.

**Caveat on the reference.** mamba-ssm cannot be installed on macOS: its
setup.py parses `torch.version.cuda`, which is None there. The reference
therefore runs `scripts/reference_gate.py`, a pure-PyTorch reimplementation of
the SSM block following mamba_ssm's own published reference semantics. Every
other part of the gate is stock PyTorch/transformers. Agreement with
mamba-ssm's CUDA kernels is untested.

**The gate's decision is not bfloat16-safe.** On a clip where p_speak sits at
0.5022 in float32, bfloat16 gives 0.4977 — the same step flips from speak to
silent. Clips whose probabilities sit far from the threshold match in either
precision. Run the gate in float32 when the decision matters.

End to end (video file to gate logits) on an M4 Max in bfloat16: about 0.8-1.0 s
for an 8-frame clip.

### Stage 4 results

`mage_vl_mlx.codec` consumes a codec asset directory (canvases +
`src_patch_position.npy` + `meta.json`) with only NumPy and PIL. Against the
official processor on three videos, the grid, **patch positions, and pixel
values are bit-identical**, and greedy 64-token output matches in float32
(logits cosine 1.000000).

codec-video-prep ships Linux-only wheels, so `docker/cv-preinfer` forwards the
binary into an ARM64 container. Point the official code at it and the codec
path runs unchanged on macOS:

```sh
docker build --platform linux/arm64 -t mage-cvprep:0.2.5 -f docker/Dockerfile.cvprep docker/
export CV_PREINFER_BIN=$PWD/docker/cv-preinfer
```

**Token efficiency.** Uniform frame sampling costs a flat 384 visual tokens per
source frame it looks at, whatever the frame budget. The codec path covered 192
of 193 source frames with 3,528 visual tokens — **18.4 tokens per covered
frame**, a 95% reduction at equal temporal coverage. Against a fixed 32-frame
uniform budget (12,288 tokens) it is a 71% reduction while seeing 6x more of the
video. On short clips sampled at only 8 frames, the codec path uses *more*
tokens (3,528 vs 3,072): the saving is in coverage per token, not absolute count.

**The streaming gate needs this path.** With frame-sampled input the gate stays
near zero on real events; with codec input it fires (soccer clip: 8 of 28
canvases above threshold, max 0.81), consistently at the last canvas of each
4-canvas group.

### Known limitations

- `scripts/generate_fixtures.py --dtype float32 --devices mps` hangs in
  PyTorch's MPS bf16→fp32 cast kernel. Generate float32 fixtures on CPU
  (~35 s per image); bfloat16 fixtures work on MPS (~6 s per image).

## Reference pins

- `microsoft/Mage` code: commit `76bec2bb3818863f470de7e867c2dc7f1d0bfd83`
- `microsoft/Mage-VL` checkpoint: revision `d88b153285f1633a61b2f693c59c8576693af185`

## Fixtures

Parity is verified against fixtures generated by the official PyTorch
implementation (greedy decoding, fixed inputs, pinned revisions). See
`scripts/generate_fixtures.py`. Generated fixtures are not committed.

On macOS the official code needs a stub `mamba_ssm` package to pass
transformers' static import check (the streaming gate is lazily imported
and unused for images): run `python scripts/install_mamba_stub.py` inside
the fixtures venv.

## License

MIT. Mage-VL itself is by Microsoft; the checkpoint is distributed under
its own license on Hugging Face.
