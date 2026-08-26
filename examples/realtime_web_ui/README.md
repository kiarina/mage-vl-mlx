# Mage-VL real-time Web UI

Local reference UI for the incremental Mage-VL API. It supports a video file
played at normal speed and a browser camera stream, while showing gate scores,
streamed text, first-token latency, full-response latency, and backlog.

The UI binds to `127.0.0.1` by default. Uploaded media and camera frames stay on
the Mac and are removed when the server exits.

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
at least as long as the decision stride.

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

This preset is an experiment, not a calibrated goal detector. The StreamMind
gate sees video only—the question does not condition its score—and earlier
tests found that `p_speak` does not reliably track event time. Start with a gate
threshold of 0 when recall matters, collect positive and negative examples,
then raise the threshold only after measuring missed events.

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

For parity with the official whole-stream behavior, the gate currently replays
the accumulated visual history for each segment. It does not yet carry Mamba
state incrementally. Long-stream backlog and the cost of this replay are
measured separately in `kiarina/labs`.
