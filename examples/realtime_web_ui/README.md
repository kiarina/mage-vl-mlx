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

### Using a phone as the camera

The same split works from a phone, but not over plain HTTP: `getUserMedia`
needs a secure context, and only `localhost` is exempt. Binding the server to
`0.0.0.0` and opening `http://<lan-ip>:8000` therefore cannot use the camera at
all — the page loads and the camera is refused.

With [Tailscale](https://tailscale.com) on both devices, `tailscale serve`
terminates TLS with a real certificate for the tailnet:

```sh
tailscale serve --bg 8000          # on the machine running the UI
# -> https://<host>.<tailnet>.ts.net/
tailscale serve --https=443 off    # to stop
```

Open that URL on the phone and the camera works. The WebSocket follows the page
scheme automatically. Pick the rear camera from the device list to point the
phone at things rather than at yourself.

The button beside the READY chip switches the viewer to fullscreen. The camera
image becomes the background, covering the screen in either orientation, and the
live response and the recent observations float over it as text rather than in
panels — older lines fade out instead of being cut off at an edge. Start/Stop
and Exit stay in the corner so a phone never has to leave the view, along with a
switch that cycles through the cameras the browser offers — front to rear on a
phone. It appears only when there is more than one to choose from, and switching
mid-run keeps the stream going.

Nothing else about the layout changes: it is the same viewer element, restyled,
and the desktop layout is untouched. Where the Fullscreen API will not take a
`div` (iOS Safari), the same mode still covers the viewport. Prefer this over exposing the port on the LAN: the UI has
no authentication, so anyone on the same Wi-Fi could otherwise drive the model,
while a tailnet is limited to your own devices. Note that enabling HTTPS for a
tailnet publishes machine names to public certificate transparency logs.

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

### Presets

The **Presets** menu loads a complete session at once: the question that defines
the labels, the labels themselves, and the sampling that suits the event.
Selecting one overwrites those fields, so edit afterwards rather than before.

| Preset | Mode | Watches for | Sampling |
|---|---|---|---|
| Defaults | Describe | — | codec, 2s stride, 4s window, 8 fps, gate 0 |
| Soccer goal | Event filter | `goal` against `none` | codec, 2s stride, 4s window, 8 fps, gate 0.3 |
| Camera gestures | Event filter | `sway`, `hand-out`, `hand-in`, `cup-in`, `cup-out` against `none` | codec, 2s stride, 4s window, 8 fps, gate 0 |

**Defaults** restores every field to the state the page loads with, which is the
way back after experimenting. The page opens on the camera tab with the codec
backend, because that is the configuration these measurements landed on: it is
the only one where live camera keeps up with the stream and the streaming gate
does anything at all.

That means the container below has to be built before the first run. **Frames
is the explicit fallback** for when Docker is not available — it runs anywhere
with no setup, at roughly three times the per-segment cost and with the gate
effectively disabled. The codec setup error says so too, so a first run without
Docker points at both the fix and the way to try it immediately.

Camera gestures is an exploration preset: it leaves the gate open so every
window reaches the VLM, keeps ignored results in the timeline, and uses a
4-second cooldown per label. Labels contain hyphens, which the first-label
normalizer keeps, so each event stays one unambiguous word.

A 2-second stride over a 4-second window means each window is analyzed twice as
it slides: finer in time than a non-overlapping stride, at twice the compute.
The window still holds 8 frames at 2 fps, which is the minimum codec accepts.
Stream lag shows how far behind the displayed result is when the stream cannot
keep up.

### Concrete example: showing only soccer goals

Choose a match video, load the **Soccer goal** preset, and start analysis. Once per
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

**The order labels appear in the question matters.** Among cases the model finds
similar, the description it reads first wins. In the gesture preset, `hand-out`
was missed while it was listed after `hand-in`; moving it above `hand-in` made
it detectable — and made it detectable at 2 fps, which raising the capture rate
to 8 fps alone had not achieved. When a label is being missed, try moving its
description earlier before reaching for more frames.

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

Capture rate is a request, not a guarantee. The browser encodes one JPEG per
frame in JavaScript and falls behind above roughly 10 fps — asking for 30 on a
MacBook Pro M1 Max delivered 10.6. Segments are written at the rate that
actually arrived, so a shortfall does not speed the motion up, and the sampling
note reports the measured rate whenever it falls short of the request.

### Keep the window at 32 frames

cv-preinfer groups frames with `--group_size 32` and emits four canvases per
group, and those canvases are the whole cost to the model. Up to 32 frames per
window there is one group, so the model's work does not change at all:

| Frames in the window | Canvases | Visual tokens | Preprocessing |
|---:|---:|---:|---:|
| 8 (2 fps × 4s) | 4 | 576 | 0.66s |
| 16 (4 fps × 4s) | 4 | 576 | 0.64s |
| **32 (8 fps × 4s)** | **4** | **576** | 0.72s |
| 64 (16 fps × 4s) | 12 | 1728 | 0.92s |
| 120 (30 fps × 4s) | 20 | 2880 | 1.21s |

So going from 2 fps to 8 fps buys four times the temporal detail for about 0.06s
— the codec analyzes finer motion and picks better patches, while the model
still sees four canvases. Past 32 frames the canvases multiply and the cost
follows: a 120 frame window took the vision tower from 0.35s to 1.21s and
generation from 1.44s to 6.81s, roughly 3.8× per segment overall.

**Aim for `context window × capture rate ≈ 32`.** A 4-second window at 8 fps is
the preset default; 2 seconds at 16 fps trades context for finer timing at the
same cost.

This applies to uploaded video too. File segments are cut at the capture rate
before codec preprocessing, so a 24 fps source does not quietly hand the codec
96 frames for a 4-second window. On a MacBook Pro M1 Max that one change took a
4-second codec segment from 8.52s to 3.11s — a real-time factor of 0.830 instead
of 2.024 — with the vision tower 3.4× faster and generation 3.2× faster.

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

## Display delay

A response can only describe a window after that window has finished, so text
always trails the picture. **Display delay** holds uploaded video back by a
chosen number of seconds while analysis starts immediately, which puts the two
back in step: set it near the FULL RESPONSE figure and a description lands on
the moment it describes rather than several seconds later.

Measured on a MacBook Pro M1 Max with a 4-second stride, a segment ending at 4s
was described while the picture showed 5.12s — 1.12s of drift instead of 4.12s
without the delay.

This changes when you see the video, not how fast the model is, so the DELAYED
badge stays on screen and STREAM LAG keeps reporting the real processing lag.

**Auto** aims at the median response time of the last few segments rather than a
number you pick. It corrects by seeking once when the gap is large and then holds
it with small playback-rate changes, so it settles within a few seconds and
follows the machine as conditions change. In a simulation of the control loop it
reached a 4 second target in 4.5 seconds and tracked a move to 1 second
immediately.

Camera mode records the stream alongside the capture that feeds the model —
VP8 in WebM, which both MediaRecorder and MediaSource accept — and plays the
picture back from that buffer, so a live camera can be held back too. The codec
backend's H.264-only requirement applies to what the model is sent, not to what
is displayed. Where the browser cannot record, the control is disabled.

It only works while the stream keeps up, which is what the **RTF** badge in the
same corner reports: seconds of work per second of video, taken as the median of
the last few segments. Below 1 the lag holds steady and a fixed delay stays in
step; at 1 or above the badge turns red, the backlog grows with every segment,
and no fixed offset can track it. Stream time, display delay and RTF stack at the
top left of the picture, and the segment state with the gate score and the three
latency figures stack at the top right, so every number can be read without
looking away from the video. Fullscreen keeps only the left column — each
observation already carries its own gate score and lag, and a phone has no room
for a second column of numbers. 

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
and lag is not read off the length of the frame queue.

Under sustained overload the lag settles rather than growing without bound. The
queue is capped and the oldest frame is dropped to make room, so the steady
state is one queue wait — the queue length divided by the capture rate — plus
one segment of processing. Measured on the frames backend with a 1-second
stride, a 4-second window, 2 fps and 640x480 input, a MacBook Pro M1 Max settles
at about 13.4s and a Mac Studio M4 Max at about 10.1s.

What is bounded is the delay, not the loss. Those two runs discarded 74% and 57%
of the frames they received, and a window nominally 4 seconds long ended up
covering 19 and 9.5 seconds of real time. Every reading stays honest about that:
the sampling note gives the measured rate, the real span and the share dropped,
the timeline stamps show the span, and the segment is written at the rate the
frames actually arrived, so a stretched window is never passed off as a dense
one.

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
