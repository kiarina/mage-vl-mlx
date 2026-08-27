const $ = (id) => document.getElementById(id);

const elements = {
  connectionDot: $("connectionDot"), systemStatus: $("systemStatus"),
  fileTab: $("fileTab"), cameraTab: $("cameraTab"),
  fileControls: $("fileControls"), cameraControls: $("cameraControls"),
  fileVideo: $("fileVideo"), cameraVideo: $("cameraVideo"), emptyStage: $("emptyStage"),
  videoInput: $("videoInput"), uploadTitle: $("uploadTitle"), uploadDetail: $("uploadDetail"),
  enableCamera: $("enableCamera"), cameraDevice: $("cameraDevice"), cameraCanvas: $("cameraCanvas"),
  analysisMode: $("analysisMode"), eventControls: $("eventControls"),
  question: $("question"), triggerLabel: $("triggerLabel"), ignoreLabel: $("ignoreLabel"),
  presetSelect: $("presetSelect"),
  cooldownSeconds: $("cooldownSeconds"), showIgnored: $("showIgnored"),
  backend: $("backend"), segmentSeconds: $("segmentSeconds"), windowSeconds: $("windowSeconds"),
  targetFps: $("targetFps"), numFrames: $("numFrames"), maxTokens: $("maxTokens"), gateThreshold: $("gateThreshold"),
  samplingNote: $("samplingNote"),
  thresholdValue: $("thresholdValue"), startButton: $("startButton"), stopButton: $("stopButton"),
  liveLabel: $("liveLabel"), liveText: $("liveText"), tokenCounter: $("tokenCounter"),
  timecode: $("timecode"), segmentState: $("segmentState"), segmentNumber: $("segmentNumber"),
  gateMetric: $("gateMetric"), firstMetric: $("firstMetric"), fullMetric: $("fullMetric"),
  lagMetric: $("lagMetric"), lagNote: $("lagNote"), timelineEntries: $("timelineEntries"),
  clearButton: $("clearButton"),
  helpDialog: $("helpDialog"), helpTitle: $("helpTitle"), helpBody: $("helpBody"), helpClose: $("helpClose"),
};

const SOCCER_QUESTION = `Classify whether this video window contains the moment a goal is scored in a soccer match.

Return exactly one lowercase label:
goal — the ball crosses the goal line and a goal is scored
none — anything else, including buildup, missed shots, replays,
or celebrations without the scoring moment

Output only goal or none.`;

const GESTURE_QUESTION = `Classify what changed in this camera window.

Return exactly one lowercase label:
sway — the person visibly rocks or sways their body or leans side to side
hand-out — a hand that was visible leaves the frame
hand-in — a hand enters the frame that was not visible before
cup-in — a cup, mug or glass enters the frame that was not visible before
cup-out — a cup, mug or glass that was visible leaves the frame
none — anything else, including no change, small shifts and talking

Report the change, not the steady state. Output only the label.`;

// Each preset is a complete session: a question that defines the labels, the
// labels themselves, and the sampling that suits the event. Labels may contain
// hyphens, which the first-label normalizer keeps, so one word per event stays
// unambiguous.
const PRESETS = {
  reset: {
    name: "↺ Defaults",
    analysisMode: "describe",
    question: "Describe what is happening. Focus on changes and motion.",
    backend: "frames",
    segmentSeconds: "4",
    windowSeconds: "4",
    targetFps: "2",
    numFrames: "16",
    gateThreshold: "0",
    maxTokens: "64",
    trigger: "goal",
    ignore: "none",
    cooldown: "8",
    showIgnored: false,
  },
  soccer: {
    name: "⚽ Soccer goal",
    analysisMode: "event",
    question: SOCCER_QUESTION,
    backend: "codec",
    segmentSeconds: "2",
    windowSeconds: "4",
    targetFps: "8",
    numFrames: "16",
    gateThreshold: "0.3",
    maxTokens: "64",
    trigger: "goal",
    ignore: "none",
    cooldown: "8",
    showIgnored: false,
  },
  gesture: {
    name: "🖐 Camera gestures",
    analysisMode: "event",
    question: GESTURE_QUESTION,
    backend: "codec",
    segmentSeconds: "2",
    windowSeconds: "4",
    targetFps: "8",
    numFrames: "16",
    // Recall matters more than saved compute while exploring, and the gate
    // scores an ordinary desk scene well below a sports broadcast, so leave it
    // open and let the labels decide.
    gateThreshold: "0",
    maxTokens: "64",
    trigger: "sway, hand-in, hand-out, cup-in, cup-out",
    ignore: "none",
    cooldown: "4",
    showIgnored: true,
  },
};

const HELP_CONTENT = {
  "analysis-mode": {
    title: "Analysis mode",
    paragraphs: [
      "Describe every response streams the model's text into Live Response for each window that passes the gate.",
      "Event filter treats the model as a short-label classifier. It buffers the answer until generation finishes, then shows only results whose first normalized label is one of the Trigger labels. Question is still required: it defines what event the model should classify.",
    ],
  },
  question: {
    title: "Question",
    paragraphs: [
      "The instruction sent to Mage-VL together with every video window. Keep it specific about what to inspect and how to answer.",
      "For Event filter, explicitly define both the trigger and non-trigger cases and require exact labels. Ask for a short answer here rather than relying on VLM max output, which truncates mid-word. The Soccer goal preset uses this complete example:",
    ],
    example: SOCCER_QUESTION,
  },
  "trigger-label": {
    title: "Trigger labels",
    paragraphs: [
      "The first labels that count as a detected event. Matching is case-insensitive after surrounding punctuation is removed.",
      "Separate several labels with commas or spaces to watch for more than one event in a single run, for example: goal, save, foul.",
      "The Question must instruct the model to return these same labels. Changing the field alone does not rewrite the Question.",
    ],
  },
  "ignore-label": {
    title: "Ignore labels",
    paragraphs: [
      "The labels for windows that do not contain a target event. These results never replace Live Response in Event filter mode.",
      "Several labels are allowed here too, separated by commas or spaces, for example: none, replay, crowd.",
      "Anything the model returns that is in neither list is shown as unmatched-label, which is how you notice the Question and the labels have drifted apart.",
    ],
  },
  cooldown: {
    title: "Cooldown",
    paragraphs: [
      "Seconds during which the same trigger label is suppressed after an accepted event. This prevents overlapping windows from reporting the same real-world event several times.",
      "Each trigger label has its own cooldown, so a run watching for several events does not let one of them hide another that happens during it.",
      "A longer cooldown merges more detections but can hide distinct events of the same kind that occur close together.",
    ],
  },
  "ignored-results": {
    title: "Ignored results",
    paragraphs: [
      "Show in timeline keeps rejected labels, gate skips, and cooldown suppressions in Observations for diagnosis. They remain excluded from Live Response.",
      "Turn this off for a clean demo; turn it on while tuning prompts, labels, and thresholds.",
    ],
  },
  backend: {
    title: "Backend",
    paragraphs: [
      "Frames samples ordinary decoded frames and works with both uploaded files and the browser camera.",
      "Codec uses Mage-VL's codec-native sparse representation through the local cv-preinfer Docker wrapper. It is available for uploaded files only and controls temporal sampling internally.",
    ],
  },
  "decision-stride": {
    title: "Decision stride",
    paragraphs: [
      "How often a new inference decision is scheduled along the stream. A 1 second stride evaluates once per second; a 4 second stride evaluates once every four seconds.",
      "Shorter strides react sooner but run the model more often and can increase backlog. The Context window can be longer than the stride, producing overlapping windows.",
    ],
  },
  "context-window": {
    title: "Context window",
    paragraphs: [
      "How much recent video each decision can inspect. For example, a 1 second stride with a 4 second context evaluates the latest four seconds once per second.",
      "Longer context can clarify an event but increases visual work and latency. It is always kept at least as long as Decision stride.",
    ],
  },
  "capture-rate": {
    title: "Capture rate",
    paragraphs: [
      "Frames sampled per second before sending a window to Mage-VL. It applies to both uploaded files and camera input with the Frames backend.",
      "Higher rates preserve faster motion but increase visual tokens, memory, and latency. Codec performs its own selection, so this control is disabled there.",
    ],
  },
  "max-frames": {
    title: "Max frames",
    paragraphs: [
      "Maximum sampled frames retained in one Frames-backend window after applying Capture rate. The effective count is the smaller of captured frames and this limit.",
      "Larger values can improve temporal coverage but substantially increase prompt length, memory, and latency. Codec does not use this setting.",
    ],
  },
  "max-output": {
    title: "VLM max output",
    paragraphs: [
      "The maximum number of new text tokens Mage-VL may generate for one window (max_new_tokens). It is a ceiling, not a required response length.",
      "Use 1–4 tokens for exact-label Event filter prompts. Descriptive questions usually need a larger limit such as 64 tokens.",
    ],
  },
  "gate-threshold": {
    title: "Gate threshold",
    paragraphs: [
      "The minimum StreamMind p(speak) score required before running VLM generation. A value of 0 sends every window to Mage-VL; higher values skip more windows.",
      "The gate sees video only and does not read Question. Its score is not a calibrated event probability, so start at 0 when recall matters and raise it only after measuring missed events.",
    ],
  },
};

let mode = "camera";
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

function openHelp(key) {
  const content = HELP_CONTENT[key];
  if (!content) return;
  elements.helpTitle.textContent = content.title;
  elements.helpBody.replaceChildren();
  content.paragraphs.forEach((text) => {
    const paragraph = document.createElement("p");
    paragraph.textContent = text;
    elements.helpBody.append(paragraph);
  });
  if (content.example) {
    const example = document.createElement("pre");
    example.className = "help-example";
    example.textContent = content.example;
    elements.helpBody.append(example);
  }
  elements.helpDialog.showModal();
}

function setMode(nextMode) {
  if (running) return;
  mode = nextMode;
  const camera = mode === "camera";
  elements.fileTab.classList.toggle("active", !camera);
  elements.cameraTab.classList.toggle("active", camera);
  elements.fileControls.classList.toggle("hidden", camera);
  elements.cameraControls.classList.toggle("hidden", !camera);
  clampGateThresholdToBackend();
  syncBackendControls();
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
    analysis_mode: elements.analysisMode.value,
    question: elements.question.value.trim(),
    backend: elements.backend.value,
    segment_s: Number(elements.segmentSeconds.value),
    window_s: Number(elements.windowSeconds.value),
    target_fps: Number(elements.targetFps.value),
    num_frames: Number(elements.numFrames.value),
    gate_threshold: Number(elements.gateThreshold.value),
    max_new_tokens: Number(elements.maxTokens.value),
    trigger_label: elements.triggerLabel.value.trim(),
    ignore_label: elements.ignoreLabel.value.trim(),
    cooldown_s: Number(elements.cooldownSeconds.value),
    show_ignored: elements.showIgnored.checked,
  };
}

function setAnalysisMode() {
  const eventMode = elements.analysisMode.value === "event";
  elements.eventControls.classList.toggle("hidden", !eventMode);
}

// The gate scores frames input near zero — measured at most 0.013 across
// sports and static scenes — so any non-zero threshold on that backend skips
// every window. Drop the threshold to 0 there and let the generated label
// decide. On codec the gate separates content types and is worth using.
const FRAMES_MAX_USEFUL_GATE = 0.002;

function clampGateThresholdToBackend() {
  if (elements.backend.value !== "frames") return;
  if (Number(elements.gateThreshold.value) <= FRAMES_MAX_USEFUL_GATE) return;
  elements.gateThreshold.value = "0";
  elements.thresholdValue.textContent = "0.00";
}

// cv-preinfer runs with --min_group_frames 8 and fails with "no canvases
// produced" below that, so a codec window needs at least this many frames.
const CODEC_MIN_FRAMES = 8;

function codecWindowFrames() {
  return Number(elements.windowSeconds.value) * Number(elements.targetFps.value);
}

// A requested capture rate is a request, not a guarantee: the browser encodes
// one JPEG per frame and falls behind at high rates. Show what actually
// arrived so a run at 30 fps is not mistaken for 30 fps of evidence.
function showMeasuredCapture(frames, fps) {
  if (mode !== "camera" || !running) return;
  const requested = Number(elements.targetFps.value);
  const short = fps < requested * 0.85;
  elements.samplingNote.textContent =
    `Capturing ${fps.toFixed(1)} fps of the ${requested} fps requested; `
    + `${frames} frames in this window.`
    + (short ? " The browser cannot keep up at this rate." : "");
  elements.samplingNote.classList.toggle("warn", short);
}

function syncBackendControls() {
  const codec = elements.backend.value === "codec";
  const camera = mode === "camera";
  // In camera mode the browser decides how many frames exist, so capture rate
  // still applies to codec: it is what fills the window.
  elements.targetFps.disabled = codec && !camera;
  elements.numFrames.disabled = codec;
  if (codec && camera) {
    const frames = codecWindowFrames();
    elements.samplingNote.textContent = frames < CODEC_MIN_FRAMES
      ? `Codec needs at least ${CODEC_MIN_FRAMES} frames per window; this capture rate and context window give ${frames}. Raise either, or switch to Frames.`
      : `The browser captures at this rate; codec sees ${frames} frames per window and samples them internally. Max frames does not apply.`;
  } else if (codec) {
    elements.samplingNote.textContent = "Codec controls temporal sampling internally; Capture rate and Max frames do not apply.";
  } else if (camera) {
    elements.samplingNote.textContent = "The browser captures the camera at this rate; each window is capped by Max frames.";
  } else {
    elements.samplingNote.textContent = "Frames backend samples the uploaded video at this rate, capped by Max frames.";
  }
  const tooFew = codec && camera && codecWindowFrames() < CODEC_MIN_FRAMES;
  elements.samplingNote.classList.toggle("warn", tooFew);
  elements.startButton.disabled = running || tooFew;
}

function keepWindowAtLeastStride() {
  const stride = Number(elements.segmentSeconds.value);
  const window = Number(elements.windowSeconds.value);
  if (window >= stride) return;
  const replacement = [...elements.windowSeconds.options]
    .map((option) => Number(option.value))
    .find((value) => value >= stride);
  elements.windowSeconds.value = String(replacement || stride);
}

function applyPreset(key) {
  const preset = PRESETS[key];
  if (running || !preset) return;
  elements.analysisMode.value = preset.analysisMode;
  elements.question.value = preset.question;
  // The gate only separates content types on codec input. With frames it never
  // rises above ~0.002, so any non-zero threshold silently suppresses every
  // window. Measured in kiarina/labs.
  elements.backend.value = preset.backend;
  elements.segmentSeconds.value = preset.segmentSeconds;
  elements.windowSeconds.value = preset.windowSeconds;
  elements.targetFps.value = preset.targetFps;
  elements.numFrames.value = preset.numFrames;
  elements.gateThreshold.value = preset.gateThreshold;
  elements.maxTokens.value = preset.maxTokens;
  elements.triggerLabel.value = preset.trigger;
  elements.ignoreLabel.value = preset.ignore;
  elements.cooldownSeconds.value = preset.cooldown;
  elements.showIgnored.checked = preset.showIgnored;
  elements.thresholdValue.textContent = Number(preset.gateThreshold).toFixed(2);
  clampGateThresholdToBackend();
  setAnalysisMode();
  syncBackendControls();
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
    if (message.effective_fps) showMeasuredCapture(message.frames, message.effective_fps);
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
  const decision = message.decision || {
    accepted: result.responded,
    visible: true,
    label: "",
    reason: result.responded ? "description" : "gate",
  };
  const firstText = metrics.first_text_s == null ? null
    : message.backlog_s + message.prepare_s + metrics.first_text_s;
  const fullResponse = message.backlog_s + message.prepare_s + metrics.total_s;
  elements.gateMetric.textContent = result.probability.toFixed(3);
  elements.firstMetric.textContent = firstText == null ? "SKIP" : `${firstText.toFixed(2)}s`;
  elements.fullMetric.textContent = metrics.generation_s == null ? "SKIP" : `${fullResponse.toFixed(2)}s`;
  elements.tokenCounter.textContent = `${metrics.generated_tokens} TOKENS`;
  elements.lagMetric.textContent = `${message.lag_s.toFixed(2)}s`;
  elements.lagNote.textContent = message.lag_s < 0.1 ? "caught up" : "behind live edge";
  elements.segmentState.textContent = decision.accepted
    ? (decision.reason === "event" ? "EVENT" : "RESPONDED")
    : decision.reason.toUpperCase();
  document.querySelector(".video-stage").classList.remove("processing");
  if (decision.accepted) elements.liveText.textContent = result.text || decision.label;
  if (!decision.visible) return;
  const empty = elements.timelineEntries.querySelector(".timeline-empty");
  if (empty) empty.remove();
  const entry = document.createElement("article");
  entry.className = "timeline-entry";
  entry.classList.toggle("event-match", decision.reason === "event");
  entry.classList.toggle("ignored", !decision.accepted);
  const text = result.text || "Gate skipped this segment.";
  const decisionText = decision.label ? `${decision.label} · ` : "";
  entry.innerHTML = `<div class="stamp">${formatTime(result.start_s)} → ${formatTime(result.end_s)}</div><p></p><div class="score">${decisionText}p ${result.probability.toFixed(3)} · lag ${message.lag_s.toFixed(2)}s</div>`;
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
elements.analysisMode.addEventListener("change", setAnalysisMode);
for (const [key, preset] of Object.entries(PRESETS)) {
  const option = document.createElement("option");
  option.value = key;
  option.textContent = preset.name;
  elements.presetSelect.append(option);
}

elements.presetSelect.addEventListener("change", (event) => {
  const key = event.target.value;
  if (!key) return;
  applyPreset(key);
  // Leave the control on its placeholder so picking the same preset again
  // re-applies it after manual edits.
  event.target.value = "";
});
elements.backend.addEventListener("change", () => { clampGateThresholdToBackend(); syncBackendControls(); });
elements.targetFps.addEventListener("change", syncBackendControls);
elements.windowSeconds.addEventListener("change", syncBackendControls);
elements.segmentSeconds.addEventListener("change", () => { keepWindowAtLeastStride(); syncBackendControls(); });
elements.gateThreshold.addEventListener("input", () => { elements.thresholdValue.textContent = Number(elements.gateThreshold.value).toFixed(2); });
elements.startButton.addEventListener("click", () => start().catch((error) => { elements.liveText.textContent = error.message; stop(); }));
elements.stopButton.addEventListener("click", stop);
elements.clearButton.addEventListener("click", clearLive);
document.querySelectorAll(".help-button").forEach((button) => {
  button.addEventListener("click", () => openHelp(button.dataset.help));
});
elements.helpClose.addEventListener("click", () => elements.helpDialog.close());
elements.helpDialog.addEventListener("click", (event) => {
  if (event.target === elements.helpDialog) elements.helpDialog.close();
});
window.addEventListener("beforeunload", () => { if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop()); });

connect();
showSource();
setAnalysisMode();
syncBackendControls();
updateClock();
