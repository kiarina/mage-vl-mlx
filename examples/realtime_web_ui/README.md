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

## Real-time semantics

A result can only begin after its input segment has completed. The displayed
first-text metric starts at that segment boundary, while stream lag reports how
far processing has fallen behind the live edge.

For parity with the official whole-stream behavior, the gate currently replays
the accumulated visual history for each segment. It does not yet carry Mamba
state incrementally. Long-stream backlog and the cost of this replay are
measured separately in `kiarina/labs`.
