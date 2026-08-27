# Mage-VL real-time Web UI

Local reference UI for the incremental Mage-VL API. It supports a video file
played at normal speed and a browser camera stream, while showing gate scores,
streamed text, first-token latency, full-response latency, and backlog.
The latest response and the scrollable observation history share the panel
below the video, so both remain visible while a stream is running.

The UI binds to `127.0.0.1` by default. Uploaded media and camera frames stay on
the Mac and are removed when the server exits.

## Where the media goes

The server binds to `127.0.0.1` and sends nothing to any external service. The
camera is captured by the browser, not by the server, so the browser and the
model do not have to run on the same machine. Forwarding the port puts the page
on `localhost` for the browser — which `getUserMedia` requires — while the model
runs elsewhere on your own network:

```sh
ssh -f -N -L 8000:127.0.0.1:8000 you@your-other-mac
```

In that setup camera frames do leave the laptop for the machine running the
model, which is why the UI says media stays on *your own machines* rather than
on this one. Nothing reaches a third party either way.

## Requirements

- Apple Silicon Mac
- Converted weights in `weights/mage-vl-bf16`
- FFmpeg and `ffprobe`
- Docker plus `docker/cv-preinfer` only when using the codec backend

## Run

From the repository root:

```sh
uv sync --group webui
uv run --group webui python examples/realtime_web_ui/app.py
```

Open <http://127.0.0.1:8000>. To use another checkpoint directory:

```sh
uv run --group webui python examples/realtime_web_ui/app.py \
  --weights /path/to/mage-vl-bf16
```

Camera mode currently uses the frames backend. File mode supports both frames
and codec. The default gate threshold is `0` so every completed segment is
described; raise it to use StreamMind as a pre-filter.

Every runtime parameter used by the reference UI can be changed before a run:
backend, decision stride, rolling context window, capture rate, maximum frames,
gate threshold, generation limit, and question. The context window is always
at least as long as the decision stride. `VLM max output` is
`max_new_tokens`: the maximum number of response tokens generated for one
window. The `?` buttons beside the controls explain their runtime and quality
tradeoffs; Question help also includes the complete soccer event-filter prompt.

Capture rate applies to both uploaded files and the browser camera when using
the frames backend. `Max frames` caps the frames retained after that temporal
sampling and can be set up to 256; large values substantially increase prompt
length, memory, and latency. The codec backend performs its own temporal and
patch selection, so both controls are disabled when codec is selected.

If the codec container or wrapper is unavailable, the UI error includes the
commands needed to start Docker Desktop, build `mage-cvprep:0.2.5`, and restart
the server. When launched from the repository root, `docker/cv-preinfer` is
detected automatically. `CV_PREINFER_BIN` remains available as an override when
the UI is launched from an installed package or another working directory. The
wrapper launches an ephemeral `docker run --rm` for each request; no persistent
container needs to stay running.

## Event filter mode

Event filter mode treats the VLM as a terse classifier and shows only a chosen
trigger label. Trigger and ignore labels, cooldown, and whether rejected
results stay visible in the diagnostic timeline are all UI controls. Responses
are buffered until classification finishes, so an ignored label never flashes
in the live response while its tokens are arriving.

The **Goal preset** provides a starting point for soccer highlights:

- 1 second decision stride and a rolling 4 second context window
- gate threshold 0.1, 2 generated tokens, `goal` / `none` labels
- 8 second cooldown to merge repeated detections of one scoring event

### Concrete example: showing only soccer goals

Choose a match video, click **Goal preset**, and start analysis. Once per
second, the UI evaluates the latest window of up to four seconds. The preset
adapts to camera mode, but see the backend note below: camera capture is
frames-only, where the gate cannot be used as a filter. The
preset supplies this question to the VLM:

```text
Classify whether this video window contains the moment a goal is scored in a
soccer match. Return exactly one lowercase label: goal if the ball crosses the
goal line and a goal is scored; none for anything else, including buildup,
missed shots, replays, or celebrations without the scoring moment. Output only
goal or none.
```

The stages have separate responsibilities:

1. StreamMind calculates `p_speak` from video only. The question does not
   affect this score. If the score is below the gate threshold, the VLM is not
   run for that window.
2. For a window that passes the gate, Mage-VL receives both the video and the
   question and generates at most two tokens.
3. Event filter buffers the response, normalizes its first label, and applies
   the UI policy below.

| VLM or gate result | UI behavior |
|---|---|
| a trigger label | Show the event unless it is inside the 8 second cooldown |
| an ignore label | Ignore it; optionally keep it in the diagnostic timeline |
| another label or sentence | Ignore it as `unmatched-label` |
| gate below threshold | Skip VLM generation and mark it as `gate` |

Both label fields accept several values separated by commas or whitespace, so
one run can watch for more than one event: `goal, save` against `none, replay`.
The cooldown is tracked per label, so a burst of one event never suppresses a
different one that happens during it.

Event filter does not replace the question: the question defines the event and
must instruct the VLM to return labels matching **Trigger labels** and **Ignore
labels**. Changing those two controls manually does not rewrite the question.

Prefer telling the model to answer briefly in the question over relying on
**VLM max output**. The cap truncates mid-token — a 16 token cap turns a camera
description into `A man is looking up and to the right, with his han` — whereas
a question that asks for one short sentence ends on its own. Keep the cap as a
latency backstop, not as the way to get short answers.

The preset uses the **codec** backend, which needs the container wrapper above.
That is not a style preference. On a clip with a goal at 6-8s, sampled in
1-second strides over a 4-second window, the two backends behave completely
differently:

| backend | gate `p_speak` range | first `goal` label |
|---|---|---|
| frames | 0.0001 - 0.0010 | 8s |
| codec | 0.62 - 0.88 | **6s** |

The gate was trained on codec-style input. With frames it never rises above
about 0.002 on soccer footage, so **any non-zero gate threshold suppresses every
window** — the preset previously shipped with frames and a threshold of 0.1 and
therefore reported nothing at all. On codec the threshold is safe anywhere from
0 to 0.7 and the label does the real work, which is why it now defaults to 0.3.

Two caveats from the same measurement. The gate does not distinguish a goal from
ordinary play — it scores content type, not events — so it is only a cheap
pre-filter. And a control clip of passing produced one `goal` label of its own,
so this is a demonstration of the event-filter mechanism rather than a
calibrated goal detector. Numbers and method are in
[`kiarina/labs`](https://github.com/kiarina/labs/blob/main/2026/08/27/mage-vl-realtime-benchmark/README.md).

For your own footage, set the gate threshold to 0 first so every window reaches
the VLM, collect positive and negative examples, then raise the threshold only
after measuring missed events. The current UI filters observations; exporting a
video clip with pre-roll and post-roll is not implemented yet.

## Camera mode and the codec backend

The browser sends independent JPEG stills, so the camera path has no compressed
stream of its own. The server assembles each segment with ffmpeg as H.264 —
not an OpenCV `mp4v` writer, whose MPEG-4 Part 2 bitstream cv-preinfer cannot
parse at all. Re-encoding stills at 2-8 fps still preserves the gate signal:
measured on a sports clip and a static scene, the gate separates them 0.79-0.82
against 0.12-0.14, against 0.001-0.013 for the same input through frames.

That makes codec the useful backend for live camera, and by a wide margin. On a
MacBook Pro M1 Max with a 4-second stride, switching the camera path from frames
to codec took a segment from about 8.5s to 2.49s — a real-time factor of 0.62
instead of 2.1 — because the visual token count drops by roughly 79% and the
generation prefill shrinks with it. Camera lag stopped growing and settled at
about 2.7 seconds.

cv-preinfer also needs at least 8 frames per window (`--min_group_frames 8`);
fewer produce `no canvases produced`. In camera mode the browser decides the
frame count, so the capture-rate control stays active for codec there and the UI
blocks Start when `context window x capture rate` falls below 8. A 2-second
window at 2 fps is 4 frames and will not run; 4 fps, or a 4-second window, does.

cv-preinfer caches its assets per video path and never evicts them. A live
stream never revisits a segment, so the UI gives the camera worker a throwaway
cache directory (`codec_cache_root` with `codec_cache_ephemeral`), which is
removed when the run stops. Without it a 1-second stride would leave roughly
2 GB per hour on disk.

When the context window is longer than the stride, windows overlap. The UI
resets gate history before each rolling window so shared frames are not appended
twice. Gate scores in that mode are independent-window scores rather than the
official whole-stream causal timeline.

## Real-time semantics

A result can only begin after its input segment has completed. The displayed
first-text metric starts at that segment boundary and includes queued backlog,
segment preparation, vision, gate, and generation prefill. Stream lag reports
how far processing has fallen behind the live edge.

Container timestamps often leave a few hundredths of a second after the final
full segment. A trailing remainder shorter than 0.5 seconds is ignored instead
of being presented as a meaningful observation.

Camera segments are timestamped by when their frames arrived, and stream lag is
measured against the newest frame in the segment just finished. Both stay
correct when the model cannot keep up: dropped frames do not rewind the clock,
and lag is not capped by the size of the frame queue. On a MacBook Pro M1 Max
with a 1-second stride, lag grows without bound, which is the honest reading —
that machine needs roughly 8 to 10 seconds per segment.

For parity with the official whole-stream behavior, the gate currently replays
the accumulated visual history for each segment. It does not yet carry Mamba
state incrementally. Long-stream backlog and the cost of this replay are
measured separately in `kiarina/labs`.

## Memory instrumentation

Two endpoints report the allocator state of the running process so a long
session can be measured without patching the UI:

```sh
curl -s http://127.0.0.1:8000/api/memory
# {"pid":..., "model_loaded":true, "active_gb":..., "cache_gb":..., "peak_gb":...}

curl -s -X POST http://127.0.0.1:8000/api/memory/reset-peak
```

`peak_gb` is cumulative for the process, so reset it before an interval you
want to attribute on its own. These numbers describe what MLX asked its
allocator for. The macOS `footprint` of the same pid is larger; the pid is
returned to let an external sampler read both at the same instant.

The gap between them is MLX's own buffer cache, not Metal overhead. The cache
holds the high-water mark of every allocation the process has made and is not
returned when a run ends, so one run with a large context window keeps that
memory reserved for as long as the process lives. Measured on a MacBook Pro
M1 Max, a light camera session settled at 22 GB of footprint, and a single run
with a 16-second window at 4 fps took it to 50 GB — while MLX reported a peak of
22 GB. **Size a machine by the footprint of the heaviest configuration you
intend to use, not by the MLX peak.**

Stopping a run now calls `mx.clear_cache()`, which drops an idle session back to
the weights (12 GB in that measurement). Clearing during a run is also safe but
does not help on its own: the same configuration re-allocates its working set
within a couple of minutes.

```sh
curl -s -X POST http://127.0.0.1:8000/api/memory/clear-cache
```

The measurements behind this are in
[`kiarina/labs`](https://github.com/kiarina/labs/blob/main/2026/08/27/mage-vl-realtime-benchmark/README.md).
