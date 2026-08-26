const $ = (id) => document.getElementById(id);

const elements = {
  connectionDot: $("connectionDot"), systemStatus: $("systemStatus"),
  fileTab: $("fileTab"), cameraTab: $("cameraTab"),
  fileControls: $("fileControls"), cameraControls: $("cameraControls"),
  fileVideo: $("fileVideo"), cameraVideo: $("cameraVideo"), emptyStage: $("emptyStage"),
  videoInput: $("videoInput"), uploadTitle: $("uploadTitle"), uploadDetail: $("uploadDetail"),
  enableCamera: $("enableCamera"), cameraDevice: $("cameraDevice"), cameraCanvas: $("cameraCanvas"),
  question: $("question"), backend: $("backend"), segmentSeconds: $("segmentSeconds"),
  targetFps: $("targetFps"), maxTokens: $("maxTokens"), gateThreshold: $("gateThreshold"),
  thresholdValue: $("thresholdValue"), startButton: $("startButton"), stopButton: $("stopButton"),
  liveLabel: $("liveLabel"), liveText: $("liveText"), tokenCounter: $("tokenCounter"),
  timecode: $("timecode"), segmentState: $("segmentState"), segmentNumber: $("segmentNumber"),
  gateMetric: $("gateMetric"), firstMetric: $("firstMetric"), fullMetric: $("fullMetric"),
  lagMetric: $("lagMetric"), lagNote: $("lagNote"), timelineEntries: $("timelineEntries"),
  clearButton: $("clearButton"),
};

let mode = "file";
let socket;
let uploaded = null;
let cameraStream = null;
let captureTimer = null;
let running = false;
let currentSegment = 0;
let streamStartedAt = 0;

function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${location.host}/ws`);
  socket.binaryType = "arraybuffer";
  socket.onopen = () => setSystem("Connected", true);
  socket.onclose = () => { setSystem("Disconnected", false); setTimeout(connect, 1200); };
  socket.onmessage = ({ data }) => handleMessage(JSON.parse(data));
}

function setSystem(text, online) {
  elements.systemStatus.textContent = text;
  elements.connectionDot.classList.toggle("online", online);
}

function setMode(nextMode) {
  if (running) return;
  mode = nextMode;
  const camera = mode === "camera";
  elements.fileTab.classList.toggle("active", !camera);
  elements.cameraTab.classList.toggle("active", camera);
  elements.fileControls.classList.toggle("hidden", camera);
  elements.cameraControls.classList.toggle("hidden", !camera);
  elements.backend.disabled = camera;
  if (camera) elements.backend.value = "frames";
  showSource();
}

function showSource() {
  const hasFile = mode === "file" && uploaded;
  const hasCamera = mode === "camera" && cameraStream;
  elements.fileVideo.classList.toggle("visible", Boolean(hasFile));
  elements.cameraVideo.classList.toggle("visible", Boolean(hasCamera));
  elements.emptyStage.classList.toggle("hidden", Boolean(hasFile || hasCamera));
}

function settings(action) {
  return {
    action,
    question: elements.question.value.trim(),
    backend: elements.backend.value,
    segment_s: Number(elements.segmentSeconds.value),
    target_fps: Number(elements.targetFps.value),
    num_frames: Math.max(8, Number(elements.segmentSeconds.value) * Number(elements.targetFps.value)),
    gate_threshold: Number(elements.gateThreshold.value),
    max_new_tokens: Number(elements.maxTokens.value),
  };
}

async function uploadVideo(file) {
  elements.uploadTitle.textContent = "Uploading…";
  elements.uploadDetail.textContent = file.name;
  const body = new FormData();
  body.append("file", file);
  try {
    const response = await fetch("/api/upload", { method: "POST", body });
    if (!response.ok) throw new Error((await response.json()).detail || "Upload failed");
    uploaded = await response.json();
    elements.fileVideo.src = uploaded.url;
    elements.uploadTitle.textContent = uploaded.name;
    elements.uploadDetail.textContent = `${formatTime(uploaded.duration)} · ready locally`;
    showSource();
  } catch (error) {
    elements.uploadTitle.textContent = "Could not load video";
    elements.uploadDetail.textContent = error.message;
  }
}

async function enableCamera() {
  if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
  const deviceId = elements.cameraDevice.value;
  cameraStream = await navigator.mediaDevices.getUserMedia({
    video: deviceId ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
                    : { width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  elements.cameraVideo.srcObject = cameraStream;
  await elements.cameraVideo.play();
  elements.enableCamera.textContent = "Camera ready";
  const devices = await navigator.mediaDevices.enumerateDevices();
  elements.cameraDevice.innerHTML = '<option value="">Default camera</option>';
  devices.filter((device) => device.kind === "videoinput").forEach((device, index) => {
    const option = document.createElement("option");
    option.value = device.deviceId;
    option.textContent = device.label || `Camera ${index + 1}`;
    elements.cameraDevice.append(option);
  });
  showSource();
}

function captureCamera() {
  const video = elements.cameraVideo;
  if (!running || socket.readyState !== WebSocket.OPEN || video.readyState < 2) return;
  const maxWidth = 768;
  const scale = Math.min(1, maxWidth / video.videoWidth);
  const canvas = elements.cameraCanvas;
  canvas.width = Math.max(2, Math.round(video.videoWidth * scale / 2) * 2);
  canvas.height = Math.max(2, Math.round(video.videoHeight * scale / 2) * 2);
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.toBlob((blob) => { if (blob && running) socket.send(blob); }, "image/jpeg", 0.84);
}

async function start() {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  clearLive();
  if (mode === "file") {
    if (!uploaded) { elements.uploadDetail.textContent = "Choose a video first"; return; }
    elements.fileVideo.currentTime = 0;
    await elements.fileVideo.play();
    socket.send(JSON.stringify({ ...settings("start_file"), media_id: uploaded.id }));
  } else {
    if (!cameraStream) await enableCamera();
    socket.send(JSON.stringify(settings("start_camera")));
    const interval = 1000 / Number(elements.targetFps.value);
    captureTimer = setInterval(captureCamera, interval);
  }
  running = true;
  streamStartedAt = performance.now();
  elements.startButton.disabled = true;
  elements.stopButton.disabled = false;
  elements.liveLabel.parentElement.classList.add("running");
  elements.liveLabel.textContent = "LIVE";
}

function stop(notify = true) {
  if (notify && socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ action: "stop" }));
  if (captureTimer) clearInterval(captureTimer);
  captureTimer = null;
  elements.fileVideo.pause();
  running = false;
  elements.startButton.disabled = false;
  elements.stopButton.disabled = true;
  elements.liveLabel.parentElement.classList.remove("running");
  elements.liveLabel.textContent = "READY";
  document.querySelector(".video-stage").classList.remove("processing");
}

function clearLive() {
  currentSegment = 0;
  elements.liveText.textContent = "Waiting for the first completed segment…";
  elements.tokenCounter.textContent = "0 TOKENS";
  elements.gateMetric.textContent = "—";
  elements.firstMetric.textContent = "—";
  elements.fullMetric.textContent = "—";
  elements.lagMetric.textContent = "—";
  elements.timelineEntries.innerHTML = '<div class="timeline-empty">Responses will appear here as the stream advances.</div>';
}

function handleMessage(message) {
  if (message.type === "model") {
    setSystem(message.state === "loading" ? "Loading 4.7B model" : "Model ready", true);
  } else if (message.type === "segment") {
    currentSegment = message.segment || currentSegment;
    elements.segmentNumber.textContent = `SEG ${String(currentSegment).padStart(2, "0")}`;
    elements.segmentState.textContent = message.state.toUpperCase();
    document.querySelector(".video-stage").classList.toggle("processing", message.state === "processing");
    if (message.state === "error") elements.liveText.textContent = message.message;
  } else if (message.type === "token") {
    elements.liveText.textContent = message.text || "…";
    elements.tokenCounter.textContent = `${message.index} TOKENS`;
  } else if (message.type === "result") {
    renderResult(message);
  } else if (message.type === "stream" && ["complete", "stopped"].includes(message.state)) {
    stop(false);
  } else if (message.type === "queue") {
    elements.lagNote.textContent = "dropping old frames";
  } else if (message.type === "error") {
    elements.liveText.textContent = message.message;
    elements.segmentState.textContent = "ERROR";
  }
}

function renderResult(message) {
  const result = message.result;
  const metrics = result.metrics;
  const firstText = metrics.first_text_s == null ? null
    : message.backlog_s + message.prepare_s + metrics.first_text_s;
  const fullResponse = message.backlog_s + message.prepare_s + metrics.total_s;
  elements.gateMetric.textContent = result.probability.toFixed(3);
  elements.firstMetric.textContent = firstText == null ? "SKIP" : `${firstText.toFixed(2)}s`;
  elements.fullMetric.textContent = metrics.generation_s == null ? "SKIP" : `${fullResponse.toFixed(2)}s`;
  elements.lagMetric.textContent = `${message.lag_s.toFixed(2)}s`;
  elements.lagNote.textContent = message.lag_s < 0.1 ? "caught up" : "behind live edge";
  elements.segmentState.textContent = result.responded ? "RESPONDED" : "SILENT";
  document.querySelector(".video-stage").classList.remove("processing");
  if (!result.responded) elements.liveText.textContent = "Gate skipped this segment.";
  const empty = elements.timelineEntries.querySelector(".timeline-empty");
  if (empty) empty.remove();
  const entry = document.createElement("article");
  entry.className = "timeline-entry";
  const text = result.text || "Gate skipped this segment.";
  entry.innerHTML = `<div class="stamp">${formatTime(result.start_s)} → ${formatTime(result.end_s)}</div><p></p><div class="score">p ${result.probability.toFixed(3)} · lag ${message.lag_s.toFixed(2)}s</div>`;
  entry.querySelector("p").textContent = text;
  elements.timelineEntries.prepend(entry);
}

function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  const rest = seconds - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${rest.toFixed(1).padStart(4, "0")}`;
}

function updateClock() {
  const seconds = mode === "file" ? elements.fileVideo.currentTime
                                   : (running ? (performance.now() - streamStartedAt) / 1000 : 0);
  elements.timecode.textContent = formatTime(seconds);
  requestAnimationFrame(updateClock);
}

elements.fileTab.addEventListener("click", () => setMode("file"));
elements.cameraTab.addEventListener("click", () => setMode("camera"));
elements.videoInput.addEventListener("change", (event) => event.target.files[0] && uploadVideo(event.target.files[0]));
elements.enableCamera.addEventListener("click", () => enableCamera().catch((error) => { elements.liveText.textContent = error.message; }));
elements.cameraDevice.addEventListener("change", () => enableCamera().catch(() => {}));
elements.gateThreshold.addEventListener("input", () => { elements.thresholdValue.textContent = Number(elements.gateThreshold.value).toFixed(2); });
elements.startButton.addEventListener("click", () => start().catch((error) => { elements.liveText.textContent = error.message; stop(); }));
elements.stopButton.addEventListener("click", stop);
elements.clearButton.addEventListener("click", clearLive);
window.addEventListener("beforeunload", () => { if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop()); });

connect();
showSource();
updateClock();
