# mage-vl-mlx

An independent MLX port of [Microsoft Mage-VL](https://huggingface.co/microsoft/Mage-VL)
(4B codec-native vision-language model) for Apple Silicon, built without
depending on mlx-vlm.

The goal is to reproduce all Mage-VL paths — static images, frame-sampled
video, proactive streaming, and the codec-native sparse video path — with
quantitative parity against the official PyTorch implementation.

## Status

Every Mage-VL path works end to end and matches the official PyTorch
implementation in float32.

| What | Verified against the official implementation |
|---|---|
| Static images | Weight keys 696/696; vision tower to 1.6e-05 relative error; greedy output identical on 3 images |
| Frame-sampled video | Preprocessing bit-identical; greedy output identical on 3 clips |
| Proactive streaming gate | Mamba mixer to 4.4e-07 (bar: 1.0e-5); speak/silent timeline identical — with one caveat about the reference, below |
| Codec-native sparse video | Selected patches and pixel values bit-identical; greedy output identical |

"Matches in float32" is the operative phrase: bfloat16 is the deployment
precision and does **not** reproduce the reference exactly. Each section below
gives the numbers and says what is not covered.

The work was done as a staged verification plan, and each stage has a lab
record with fixtures, failed attempts, and measurements, in
[kiarina/labs](https://github.com/kiarina/labs/blob/main/docs/mage-vl-mlx-port.md).

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
decision flips under bfloat16 when a score sits near the threshold.

`--segment-sec` is the decision interval: the gate emits one speak/silent
decision per segment. It works down to 1s with either backend. A trailing
segment shorter than a codec group cannot be preprocessed and is skipped, the
way the official script skips unusable segments.

### Real-time API and local Web UI

`mage_vl_mlx.realtime.RealtimeSession` accepts one completed segment at a time,
streams generated tokens through a callback, and reports preprocessing,
vision, gate, first-token, full-generation, and peak-memory measurements. The
model can stay in bfloat16 while the threshold-sensitive gate runs in float32.

The reference Web UI plays a video at normal speed or captures the Mac camera,
then shows the generated text, gate score, latency, and backlog beside the live
image. It binds to localhost and sends no media to an external service.

```sh
uv sync --group webui
uv run --group webui python examples/realtime_web_ui/app.py
# open http://127.0.0.1:8000
```

File mode supports the frames and codec backends. Camera mode currently uses
sampled frames. See [`examples/realtime_web_ui`](examples/realtime_web_ui) for
the controls, privacy boundary, and real-time semantics.

Measure processing-only real-time factor and simulated backlog without the UI:

```sh
uv run python scripts/benchmark_realtime.py \
  --video clip.mp4 --segment-sec 4 --gate-threshold 0
```

The gate currently replays all accumulated visual history for every segment.
This matches the official whole-stream result, but it is not yet a stateful
incremental Mamba implementation. The long-stream cost is therefore reported
as a measured limitation rather than hidden by a rolling window.
Final container-duration slivers shorter than 0.5 seconds are ignored by the
benchmark and Web UI rather than treated as standalone observations.

### Using this for event detection

The gate is not an event detector. It separates content types — a sports
broadcast scores ~0.7-0.8, a quiet hallway ~0.05-0.11 — but its score does not
reliably track *when* something happens. On a clip of a glass falling at ~6-7s,
sampled at 1s segments, the scores were:

```
0.02  0.52  0.14  0.49  0.45  0.07  0.50  0.74
                                      ^^^^  ^^^^  glass falls here
```

The event segments (0.50, 0.74) are barely separable from a segment where
nothing happens (0.52), and on other clips the event scores *below* its own
quiet segments.

**The generated text discriminates far better than the score.** Same run, same
segments:

```
1-2s (0.52): "The scene remains static ... no noticeable changes or movements."
3-4s (0.49): "The scene remains static ... no noticeable changes or movements."
6-7s (0.50): "Suddenly, a clear glass container starts to move, tilting and
              rotating in an unusual manner."
7-8s (0.74): "a glass object is seen moving rapidly, creating a blurred effect"
```

So the practical recipe is to use the gate only as a cheap pre-filter and read
the text for the actual decision:

```sh
python inference_streaming.py --video clip.mp4 --segment-sec 1 \
  --gate-threshold 0.3          # or 0 to describe every segment
```

Object identity wanders between segments (the same glass is described as a
pitcher, then a glass with a spoon), but the static-versus-moving distinction
stays consistent — which is the part event detection needs.

Prompt building (`mage_vl_mlx.prompt`) uses `tokenizers` and `jinja2`, neither
of which pulls in torch. Its token ids match the official processor exactly for
images, frame-sampled video, and codec video.

## Verification

Numbers below come from comparing against the official PyTorch implementation
on an M4 Max. Fixtures are generated in float32 on CPU; see `scripts/check_*.py`.

### Static images

[Lab record](https://github.com/kiarina/labs/blob/main/2026/08/25/mage-vl-mlx-stage1-image-parity/README.md) — fixtures, failed attempts, full measurements.

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

### Frame-sampled video

[Lab record](https://github.com/kiarina/labs/blob/main/2026/08/25/mage-vl-mlx-stage2-video-parity/README.md) — fixtures, failed attempts, full measurements.

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

### Proactive streaming gate

[Lab record](https://github.com/kiarina/labs/blob/main/2026/08/25/mage-vl-mlx-stage3-streaming-gate/README.md) — fixtures, failed attempts, full measurements.

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

### Codec-native sparse video

[Lab record](https://github.com/kiarina/labs/blob/main/2026/08/26/mage-vl-mlx-stage4-codec-native/README.md) — fixtures, failed attempts, full measurements.

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

## Known limitations

- **The model ignores system messages.** The chat template supports one — pass
  it through `PromptBuilder.render()` — but the checkpoint does not follow it.
  "Always begin every reply with the word BANANA", "You are a pirate", and a
  request to answer in Japanese were all ignored, with output identical to the
  default system turn. The CLI deliberately exposes no `--system` flag. Steer
  through `--question` instead, which does change the output.
- No quantization: bfloat16 only, so this is slower than an 8-bit checkpoint.
- The real-time gate replays accumulated visual history instead of carrying
  Mamba state incrementally; backlog can grow on long streams.
- No test suite; verification runs through `scripts/check_*.py` by hand.
- `scripts/generate_fixtures.py --dtype float32 --devices mps` hangs in
  PyTorch's MPS bf16→fp32 cast kernel. Generate float32 fixtures on CPU
  (~35 s per image); bfloat16 fixtures work on MPS (~6 s per image).

## Repository layout

Running the model needs none of the verification machinery — `src/` and the two
inference scripts are torch-free.

| Path | Needs | Purpose |
|---|---|---|
| `src/mage_vl_mlx/` | mlx, numpy, pillow, opencv, tokenizers, jinja2 | The port: vision tower, decoder, video and codec preprocessing, streaming gate, prompt building, real-time session API |
| `inference_base.py`, `inference_streaming.py` | same | Run the model |
| `examples/realtime_web_ui/` | **`--group webui`** | Local video-file and camera reference UI |
| `scripts/convert_weights.py`, `convert_gate_weights.py` | same | Convert the checkpoint once, into `weights/` |
| `scripts/benchmark.py`, `benchmark_realtime.py`, `gate_stream.py`, `gate_timeline.py` | same | Measurement and inspection |
| `scripts/generate_*_fixtures.py`, `check_codec_parity.py`, `debug_vision.py`, `reference_gate.py` | **`--group fixtures`** (torch, transformers) | Produce reference outputs from the official implementation and compare against them |
| `docker/` | Docker | Runs the Linux-only cv-preinfer for the codec path |

Install the verification dependencies only if you want to re-run the parity
checks: `uv sync --group fixtures`.

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

## License and attribution

This port is MIT (see `LICENSE`). It contains no upstream source and
redistributes no model weights — everything here was written against the
public implementations and papers listed below.

| Upstream | License | How this repo relates to it |
|---|---|---|
| [microsoft/Mage](https://github.com/microsoft/Mage) | MIT | The architecture, preprocessing rules, and the CLI of `inference_base.py` / `inference_streaming.py` were reimplemented by reading `mage_vl/` at commit `76bec2bb3818863f470de7e867c2dc7f1d0bfd83`. |
| [microsoft/Mage-VL](https://huggingface.co/microsoft/Mage-VL) checkpoint | Apache-2.0 | Not redistributed. `scripts/convert_weights.py` downloads it from Hugging Face and converts it locally; you accept the model's own terms there. |
| [state-spaces/mamba](https://github.com/state-spaces/mamba) | Apache-2.0 | `scripts/reference_gate.py` reimplements the Mamba1 selective scan in plain PyTorch, following that project's published reference semantics (`selective_scan_ref` and the non-fast path of `mamba_simple.Mamba.forward`), because mamba-ssm has no macOS build. |
| [huggingface/transformers](https://github.com/huggingface/transformers) | Apache-2.0 | The Qwen2-VL image preprocessing rules — smart_resize, the fused rescale/normalize order, and the patch layout — were matched against `Qwen2VLImageProcessor` so that preprocessing is bit-identical. |

One caveat worth knowing before you depend on this: `codec-video-prep`, the
package `docker/Dockerfile.cvprep` installs to run the codec path, **declares no
license** in its PyPI metadata and publishes no source. It is Microsoft's own
dependency for that path, but if licensing matters for your use, check it
before shipping anything built on the codec backend. The image is built
locally and is not published anywhere.

Mage-VL is Microsoft's work; this repository only ports it to MLX.
