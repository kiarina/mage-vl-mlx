"""FastAPI backend for the local Mage-VL real-time reference UI."""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import queue
import re
import shutil
import tempfile
import threading
import time
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import mlx.core as mx

from mage_vl_mlx.realtime import (
    RealtimeSession,
    extract_subclip,
    video_duration,
)

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
UPLOAD_ROOT = Path(tempfile.mkdtemp(prefix="mage-vl-webui-"))
UPLOADS: dict[str, Path] = {}


class ModelEngine:
    """One local model instance shared by the single active demo stream."""

    def __init__(self, weights: Path):
        self.weights = weights
        self.lock = threading.Lock()
        self.template: RealtimeSession | None = None

    def load(self) -> None:
        if self.template is None:
            self.template = RealtimeSession.from_pretrained(
                self.weights,
                model_dtype=mx.bfloat16,
                gate_dtype=mx.float32,
                video_backend="frames",
            )

    def create_session(self, settings: dict) -> RealtimeSession:
        self.load()
        assert self.template is not None
        return RealtimeSession(
            self.template.model,
            self.template.gate,
            self.template.prompt_builder,
            model_dtype=mx.bfloat16,
            gate_dtype=mx.float32,
            video_backend=settings["backend"],
            num_frames=settings["num_frames"],
            target_fps=settings["target_fps"],
            gate_threshold=settings["gate_threshold"],
            max_new_tokens=settings["max_new_tokens"],
        )


engine = ModelEngine(Path(os.environ.get("MAGE_VL_WEIGHTS", "weights/mage-vl-bf16")))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    shutil.rmtree(UPLOAD_ROOT, ignore_errors=True)


app = FastAPI(
    title="Mage-VL Live Vision", docs_url=None, redoc_url=None, lifespan=lifespan
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def settings_from(message: dict) -> dict:
    backend = str(message.get("backend", "frames"))
    if backend not in {"frames", "codec"}:
        raise ValueError("backend must be frames or codec")
    analysis_mode = str(message.get("analysis_mode", "describe"))
    if analysis_mode not in {"describe", "event"}:
        raise ValueError("analysis_mode must be describe or event")
    segment_s = min(8.0, max(0.5, float(message.get("segment_s", 4.0))))
    window_s = min(16.0, max(segment_s, float(message.get("window_s", segment_s))))
    return {
        "backend": backend,
        "analysis_mode": analysis_mode,
        "question": str(message.get("question") or "Describe what is happening."),
        "segment_s": segment_s,
        "window_s": window_s,
        "target_fps": min(8.0, max(0.5, float(message.get("target_fps", 2.0)))),
        "num_frames": min(256, max(1, int(message.get("num_frames", 16)))),
        "gate_threshold": min(1.0, max(0.0, float(message.get("gate_threshold", 0.0)))),
        "max_new_tokens": min(256, max(1, int(message.get("max_new_tokens", 80)))),
        "trigger_label": str(message.get("trigger_label") or "goal").strip().lower(),
        "ignore_label": str(message.get("ignore_label") or "none").strip().lower(),
        "cooldown_s": min(120.0, max(0.0, float(message.get("cooldown_s", 8.0)))),
        "show_ignored": bool(message.get("show_ignored", False)),
    }


def normalized_label(text: str) -> str:
    """Return a robust first label from a terse classifier response."""
    labels = re.findall(r"[\w-]+", text.lower(), flags=re.UNICODE)
    return labels[0] if labels else ""


def result_decision(
    result: dict,
    settings: dict,
    *,
    end_s: float,
    last_event_s: float | None,
) -> dict:
    """Apply UI response filtering without changing the model or gate result."""
    if not result["responded"]:
        return {"accepted": False, "visible": settings["show_ignored"],
                "label": "", "reason": "gate"}
    if settings["analysis_mode"] == "describe":
        return {"accepted": True, "visible": True, "label": "", "reason": "description"}

    label = normalized_label(result["text"])
    if label == settings["ignore_label"]:
        reason = "ignored-label"
        accepted = False
    elif label != settings["trigger_label"]:
        reason = "unmatched-label"
        accepted = False
    elif last_event_s is not None and end_s - last_event_s < settings["cooldown_s"]:
        reason = "cooldown"
        accepted = False
    else:
        reason = "event"
        accepted = True
    return {
        "accepted": accepted,
        "visible": accepted or settings["show_ignored"],
        "label": label,
        "reason": reason,
    }


def camera_clip(images: list[bytes], fps: float, output: Path) -> None:
    import cv2
    import numpy as np

    frames = []
    for encoded in images:
        frame = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            frames.append(frame)
    if not frames:
        raise ValueError("no camera frames could be decoded")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("could not create camera segment")
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "model_loaded": engine.template is not None}


@app.get("/api/memory")
def memory():
    """Report MLX allocator state so a sampler can correlate it with the OS view.

    MLX reports what this process asked the allocator for. macOS `footprint`
    additionally counts Metal's reserved and cached device memory, so the two
    numbers are not interchangeable. The pid is returned to let an external
    sampler read RSS and footprint at the same instant.
    """
    return {
        "pid": os.getpid(),
        "model_loaded": engine.template is not None,
        "active_gb": mx.get_active_memory() / 1024**3,
        "cache_gb": mx.get_cache_memory() / 1024**3,
        "peak_gb": mx.get_peak_memory() / 1024**3,
    }


@app.post("/api/memory/reset-peak")
def reset_peak_memory():
    """Reset the MLX peak counter so the next interval is measured on its own."""
    mx.reset_peak_memory()
    return {"ok": True, "peak_gb": mx.get_peak_memory() / 1024**3}


@app.post("/api/memory/clear-cache")
def clear_cache():
    """Return MLX's buffer cache to the system.

    The cache holds the high-water mark of every allocation the process has made,
    so one run with a large window keeps that memory reserved for the lifetime of
    the process. Clearing costs re-allocation on the next segment.
    """
    before = mx.get_cache_memory()
    mx.clear_cache()
    return {
        "ok": True,
        "freed_gb": (before - mx.get_cache_memory()) / 1024**3,
        "cache_gb": mx.get_cache_memory() / 1024**3,
    }


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".m4v", ".webm", ".avi"}:
        raise HTTPException(status_code=400, detail="unsupported video format")
    media_id = uuid.uuid4().hex
    path = UPLOAD_ROOT / f"{media_id}{suffix}"
    with path.open("wb") as output:
        while chunk := await file.read(1024 * 1024):
            output.write(chunk)
    try:
        duration = video_duration(path)
    except Exception as error:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="video could not be read") from error
    UPLOADS[media_id] = path
    return {
        "id": media_id,
        "name": file.filename,
        "duration": duration,
        "url": f"/media/{media_id}",
    }


@app.get("/media/{media_id}")
def media(media_id: str):
    path = UPLOADS.get(media_id)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="media not found")
    return FileResponse(path)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()
    send_lock = asyncio.Lock()
    stop = threading.Event()
    camera_frames: queue.Queue[tuple[float, bytes]] | None = None
    worker: asyncio.Task | None = None

    async def send(kind: str, **data) -> None:
        async with send_lock:
            await websocket.send_json({"type": kind, **data})

    def emit(kind: str, **data) -> None:
        asyncio.run_coroutine_threadsafe(send(kind, **data), loop)

    def token_callback(segment: int):
        def callback(token: int, text: str, index: int, elapsed: float) -> None:
            emit(
                "token",
                segment=segment,
                token=token,
                index=index,
                text=text,
                elapsed_s=elapsed,
            )
        return callback

    def process_file(media_id: str, settings: dict) -> None:
        source = UPLOADS.get(media_id)
        if source is None:
            emit("error", message="Uploaded video is no longer available.")
            return
        with engine.lock:
            emit("model", state="loading")
            session = engine.create_session(settings)
            emit("model", state="ready")
            duration = video_duration(source)
            baseline = time.perf_counter()
            segment_index = 0
            last_event_s: float | None = None
            with tempfile.TemporaryDirectory(dir=source.parent) as directory:
                previous_boundary_s = 0.0
                boundary_s = settings["segment_s"]
                while previous_boundary_s < duration - 1e-3 and not stop.is_set():
                    end_s = min(duration, boundary_s)
                    if previous_boundary_s > 0 and end_s - previous_boundary_s < 0.5:
                        break
                    start_s = max(0.0, end_s - settings["window_s"])
                    wait_s = baseline + end_s - time.perf_counter()
                    if wait_s > 0 and stop.wait(wait_s):
                        break
                    ready_at = baseline + end_s
                    backlog_at_start = max(0.0, time.perf_counter() - ready_at)
                    segment_index += 1
                    emit(
                        "segment",
                        state="processing",
                        segment=segment_index,
                        start_s=start_s,
                        end_s=end_s,
                        backlog_s=backlog_at_start,
                    )
                    clip = Path(directory) / f"segment-{segment_index:04d}.mp4"
                    prepare_start = time.perf_counter()
                    try:
                        extract_subclip(source, start_s, end_s - start_s, clip)
                        prepare_s = time.perf_counter() - prepare_start
                        if settings["window_s"] > settings["segment_s"]:
                            # Overlapping windows must not duplicate their shared frames
                            # in the causal gate history. Evaluate each rolling window alone.
                            session.reset()
                        result = session.process_segment(
                            clip,
                            settings["question"],
                            start_s=start_s,
                            end_s=end_s,
                            on_token=(
                                token_callback(segment_index)
                                if settings["analysis_mode"] == "describe"
                                else None
                            ),
                        )
                    except Exception as error:
                        emit(
                            "segment",
                            state="error",
                            segment=segment_index,
                            message=f"{type(error).__name__}: {error}",
                        )
                        previous_boundary_s = end_s
                        boundary_s += settings["segment_s"]
                        continue
                    lag_s = max(0.0, time.perf_counter() - ready_at)
                    rendered = result.to_dict()
                    decision = result_decision(
                        rendered,
                        settings,
                        end_s=end_s,
                        last_event_s=last_event_s,
                    )
                    if decision["accepted"]:
                        last_event_s = end_s
                    emit(
                        "result",
                        segment=segment_index,
                        result=rendered,
                        decision=decision,
                        prepare_s=prepare_s,
                        backlog_s=backlog_at_start,
                        lag_s=lag_s,
                    )
                    previous_boundary_s = end_s
                    boundary_s += settings["segment_s"]
            emit("stream", state="stopped" if stop.is_set() else "complete")

    def process_camera(
        frame_queue: queue.Queue[tuple[float, bytes]], settings: dict
    ) -> None:
        with engine.lock:
            emit("model", state="loading")
            session = engine.create_session(settings)
            emit("model", state="ready")
            frames_per_stride = max(
                1, round(settings["segment_s"] * settings["target_fps"])
            )
            frames_per_window = max(
                frames_per_stride,
                round(settings["window_s"] * settings["target_fps"]),
            )
            segment_index = 0
            origin: float | None = None
            frames_since_segment = 0
            captured: deque[tuple[float, bytes]] = deque(maxlen=frames_per_window)
            last_event_s: float | None = None
            with tempfile.TemporaryDirectory() as directory:
                while not stop.is_set():
                    try:
                        captured.append(frame_queue.get(timeout=0.25))
                        frames_since_segment += 1
                    except queue.Empty:
                        continue
                    if origin is None:
                        origin = captured[0][0]
                    if frames_since_segment < frames_per_stride:
                        continue
                    frames_since_segment = 0
                    segment_index += 1
                    # Camera segments are timestamped by when their frames arrived,
                    # not by counting processed segments. When the model falls behind
                    # and old frames are dropped, this keeps the displayed times equal
                    # to real time instead of drifting by the number of dropped frames.
                    newest_at = captured[-1][0]
                    start_s = captured[0][0] - origin
                    end_s = newest_at - origin
                    clip = Path(directory) / f"camera-{segment_index:04d}.mp4"
                    prepare_start = time.perf_counter()
                    try:
                        camera_clip(
                            [frame for _, frame in captured],
                            settings["target_fps"],
                            clip,
                        )
                        prepare_s = time.perf_counter() - prepare_start
                        backlog = time.perf_counter() - newest_at
                        emit(
                            "segment",
                            state="processing",
                            segment=segment_index,
                            start_s=start_s,
                            end_s=end_s,
                            backlog_s=backlog,
                        )
                        if settings["window_s"] > settings["segment_s"]:
                            session.reset()
                        result = session.process_segment(
                            clip,
                            settings["question"],
                            start_s=start_s,
                            end_s=end_s,
                            on_token=(
                                token_callback(segment_index)
                                if settings["analysis_mode"] == "describe"
                                else None
                            ),
                        )
                        rendered = result.to_dict()
                        decision = result_decision(
                            rendered,
                            settings,
                            end_s=end_s,
                            last_event_s=last_event_s,
                        )
                        if decision["accepted"]:
                            last_event_s = end_s
                        emit(
                            "result",
                            segment=segment_index,
                            result=rendered,
                            decision=decision,
                            prepare_s=prepare_s,
                            backlog_s=backlog,
                            lag_s=time.perf_counter() - newest_at,
                        )
                    except Exception as error:
                        emit(
                            "segment",
                            state="error",
                            segment=segment_index,
                            message=f"{type(error).__name__}: {error}",
                        )
            emit("stream", state="stopped")

    async def stop_worker() -> None:
        nonlocal worker, camera_frames
        stop.set()
        if worker is not None:
            try:
                await worker
            except asyncio.CancelledError:
                pass
            worker = None
            # MLX keeps the high-water mark of every allocation for the life of
            # the process. Without this, an idle session that once ran a large
            # window keeps tens of GB reserved. The next run re-allocates its own
            # working set within the first few segments.
            mx.clear_cache()
        camera_frames = None

    try:
        await send("connected")
        while True:
            incoming = await websocket.receive()
            if incoming.get("bytes") is not None:
                if camera_frames is not None:
                    stamped = (time.perf_counter(), incoming["bytes"])
                    try:
                        camera_frames.put_nowait(stamped)
                    except queue.Full:
                        try:
                            camera_frames.get_nowait()
                        except queue.Empty:
                            pass
                        camera_frames.put_nowait(stamped)
                        await send("queue", state="dropping", frames=camera_frames.qsize())
                continue
            text = incoming.get("text")
            if text is None:
                break
            message = json.loads(text)
            action = message.get("action")
            if action == "stop":
                await stop_worker()
                await send("stream", state="stopped")
            elif action == "start_file":
                await stop_worker()
                stop.clear()
                settings = settings_from(message)
                worker = asyncio.create_task(asyncio.to_thread(
                    process_file, str(message["media_id"]), settings
                ))
                await send("stream", state="running", mode="file")
            elif action == "start_camera":
                await stop_worker()
                stop.clear()
                settings = settings_from(message)
                if settings["backend"] != "frames":
                    await send(
                        "error",
                        message="Camera mode currently uses the frames backend.",
                    )
                    continue
                camera_frames = queue.Queue(maxsize=max(
                    16, round(settings["target_fps"] * settings["segment_s"] * 4)
                ))
                worker = asyncio.create_task(asyncio.to_thread(
                    process_camera, camera_frames, settings
                ))
                await send("stream", state="running", mode="camera")
            else:
                await send("error", message=f"Unknown action: {action}")
    except Exception:
        stop.set()
    finally:
        await stop_worker()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("weights/mage-vl-bf16"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    global engine
    engine = ModelEngine(args.weights)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
