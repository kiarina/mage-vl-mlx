const $ = (id) => document.getElementById(id);

const elements = {
  connectionDot: $("connectionDot"), systemStatus: $("systemStatus"),
  fileTab: $("fileTab"), cameraTab: $("cameraTab"),
  fileControls: $("fileControls"), cameraControls: $("cameraControls"),
  fileVideo: $("fileVideo"), cameraVideo: $("cameraVideo"), emptyStage: $("emptyStage"),
  videoInput: $("videoInput"), uploadTitle: $("uploadTitle"), uploadDetail: $("uploadDetail"),
  cameraDevice: $("cameraDevice"), cameraCanvas: $("cameraCanvas"),
  analysisMode: $("analysisMode"), eventControls: $("eventControls"),
  question: $("question"), triggerLabel: $("triggerLabel"), ignoreLabel: $("ignoreLabel"),
  presetSelect: $("presetSelect"),
  displayDelay: $("displayDelay"), delayBadge: $("delayBadge"), delayValue: $("delayValue"),
  rtfBadge: $("rtfBadge"), rtfValue: $("rtfValue"),
  cameraDelayed: $("cameraDelayed"), pipLabel: $("pipLabel"),
  stageTitle: $("stageTitle"), stageNote: $("stageNote"),
  viewerCard: document.querySelector(".viewer-card"),
  immersiveButton: $("immersiveButton"), immersiveToggle: $("immersiveToggle"),
  immersiveExit: $("immersiveExit"), immersiveCamera: $("immersiveCamera"),
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
    backend: "codec",
    segmentSeconds: "2",
    windowSeconds: "4",
    // 4s at 8 fps is 32 frames: the most temporal detail cv-preinfer will take
    // before it starts emitting extra canvases and the model's work grows.
    targetFps: "8",
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
  "display-delay": {
    title: "Display delay",
    paragraphs: [
      "Holds the picture back by this many seconds while analysis starts immediately, so a response lands on the moment it describes instead of trailing it. Match it to the FULL RESPONSE figure below the viewer: at a 2 second response, a 2 second delay puts text and picture in step.",
      "This changes when you see the video, not how fast the model is. The DELAYED badge stays on screen so the offset is never mistaken for the processing time, which STREAM LAG keeps reporting.",
      "Auto aims at the median response time of the last few segments instead of a fixed number, and reaches it by playing slightly slow or slightly fast until the offset matches. It needs no guess up front and follows the machine as conditions change.",
      "Camera mode records the stream alongside the capture that feeds the model and plays it back from that buffer, so a live camera can be held back too. Where the browser cannot record, the control is disabled.",
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

let mode = "file";
let delayTimer = null;
let recentWork = [];
let recentResponse = [];
let cameraFrameCounts = { received: 0, dropped: 0 };
let dvr = null;
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

// The tab is the switch. Choosing Camera is the gesture that asks for
// permission, and leaving it releases the device: a camera has no business
// staying on while a file is being watched, and a separate button for it read
// as a status while quietly restarting the stream.
function setMode(nextMode) {
  if (running) return;
  mode = nextMode;
  const camera = mode === "camera";
  if (camera) {
    enableCamera().catch((error) => {
      elements.liveText.textContent = `Could not start the camera: ${error.message}`;
      setMode("file");
    });
  } else {
    releaseCamera();
  }
  elements.fileTab.classList.toggle("active", !camera);
  elements.cameraTab.classList.toggle("active", camera);
  elements.fileControls.classList.toggle("hidden", camera);
  elements.cameraControls.classList.toggle("hidden", !camera);
  clampGateThresholdToBackend();
  syncBackendControls();
  syncImmersiveControls();
  syncDelayBadge();
  showSource();
}

// Immersive mode reuses the viewer card rather than duplicating it: the video,
// the live response and the timeline already live there, so the Fullscreen API
// plus one class is the whole feature. The class also works alone, which is the
// fallback where requestFullscreen is not available for a div.
function immersiveActive() {
  return elements.viewerCard.classList.contains("immersive");
}

// The device list only carries real entries once permission has been granted,
// so the switch appears when there is actually something to switch between.
function cameraOptions() {
  return [...elements.cameraDevice.options].filter((option) => option.value);
}

function syncImmersiveControls() {
  const active = immersiveActive();
  elements.immersiveButton.setAttribute("aria-pressed", String(active));
  elements.immersiveButton.setAttribute(
    "aria-label", active ? "Exit fullscreen" : "Enter fullscreen");
  elements.immersiveToggle.textContent = running ? "Stop" : "Start";
  elements.immersiveToggle.classList.toggle("primary", !running);
  const options = cameraOptions();
  elements.immersiveCamera.classList.toggle(
    "hidden", mode !== "camera" || options.length < 2);
  const current = options.find((option) => option.value === elements.cameraDevice.value);
  elements.immersiveCamera.title = current
    ? `Switch camera (now: ${current.textContent})`
    : "Switch camera";
}

async function cycleCamera() {
  const options = cameraOptions();
  if (options.length < 2) return;
  const index = options.findIndex((option) => option.value === elements.cameraDevice.value);
  // An unset value means the browser default, which is one of these devices;
  // stepping from -1 lands on the first named one.
  const next = options[(index + 1) % options.length];
  elements.cameraDevice.value = next.value;
  // Label the button for the device now selected before awaiting the stream,
  // so the control never describes the camera it just left.
  syncImmersiveControls();
  try {
    await enableCamera();
  } catch (error) {
    elements.liveText.textContent = `Could not switch camera: ${error.message}`;
  }
}

function setImmersive(active) {
  elements.viewerCard.classList.toggle("immersive", active);
  syncImmersiveControls();
  if (active && document.fullscreenElement !== elements.viewerCard) {
    elements.viewerCard.requestFullscreen?.().catch(() => {});
  } else if (!active && document.fullscreenElement) {
    document.exitFullscreen?.().catch(() => {});
  }
}

// A live camera has nothing to rewind, so one is recorded alongside the capture
// that feeds the model and played back from a buffer. VP8 in WebM is the pairing
// both MediaRecorder and MediaSource accept; the codec backend's H.264-only
// requirement applies to what the model is sent, not to what is displayed.
const DVR_MIME = ["video/webm;codecs=vp8", "video/webm;codecs=vp9"].find(
  (type) => window.MediaRecorder?.isTypeSupported(type) && window.MediaSource?.isTypeSupported(type));

function startDvr(stream) {
  if (!DVR_MIME) return null;
  // The recorder is built first because it is the part that refuses: a stream
  // whose tracks have ended throws here. Doing it before the element is pointed
  // at a MediaSource keeps a refusal from leaving a dead source on the stage.
  let recorder;
  try {
    recorder = new MediaRecorder(stream, { mimeType: DVR_MIME });
  } catch (_) {
    return null;
  }
  const source = new MediaSource();
  const state = { source, recorder, buffer: null, queue: [], pump: null };
  elements.cameraDelayed.src = URL.createObjectURL(source);
  // Until a frame decodes there is nothing to put on the stage, so the handover
  // waits for one rather than blacking it out.
  elements.cameraDelayed.addEventListener("loadeddata", () => {
    if (dvr !== state) return;
    elements.cameraDelayed.play().catch(() => {});
    showSource();
  }, { once: true });
  source.addEventListener("sourceopen", () => {
    state.buffer = source.addSourceBuffer(DVR_MIME);
    // Old history is left to the browser's own coded frame eviction. Removing
    // it here emptied the buffer instead of trimming it: a removal runs to the
    // next random access point, and this recording carries few keyframes.
    state.pump = setInterval(() => {
      if (state.buffer && !state.buffer.updating && state.queue.length) {
        try { state.buffer.appendBuffer(state.queue.shift()); } catch (_) { state.queue.length = 0; }
      }
    }, 60);
  }, { once: true });
  state.recorder.ondataavailable = async (event) => {
    if (event.data.size) state.queue.push(new Uint8Array(await event.data.arrayBuffer()));
  };
  state.recorder.start(200);
  return state;
}

// The live picture is what a camera is aimed with, so a delayed stage keeps it
// as an inset instead of replacing it. The element is the same one the capture
// loop reads frames from, which is why it was only ever hidden, never stopped.
function showLiveInset(on) {
  elements.cameraVideo.classList.toggle("pip", on);
  elements.pipLabel.classList.toggle("hidden", !on);
}

// The inset is only a preview if it is shaped like the stage: a fixed ratio
// crops a portrait phone's picture to a landscape strip, which shows a framing
// nobody is recording. Width is bounded on both axes so a tall stage cannot
// produce an inset that runs down the screen.
function syncInsetShape() {
  const stage = document.querySelector(".video-stage");
  const width = stage.clientWidth;
  const height = stage.clientHeight;
  if (!width || !height) return;
  const byWidth = Math.min(Math.max(width * 0.21, 112), 224);
  const byHeight = height * 0.3 * (width / height);
  stage.style.setProperty("--pip-aspect", `${width} / ${height}`);
  stage.style.setProperty("--pip-w", `${Math.round(Math.min(byWidth, byHeight))}px`);
}

function stopDvr() {
  if (!dvr) return;
  if (dvr.pump) clearInterval(dvr.pump);
  try { dvr.recorder?.state !== "inactive" && dvr.recorder.stop(); } catch (_) {}
  try { dvr.source.readyState === "open" && dvr.source.endOfStream(); } catch (_) {}
  elements.cameraDelayed.removeAttribute("src");
  elements.cameraDelayed.load();
  elements.cameraDelayed.classList.remove("visible");
  showLiveInset(false);
  dvr = null;
}

function delayIsAuto() {
  return elements.displayDelay.value === "auto";
}

function targetDelaySeconds() {
  if (!delayIsAuto()) return Number(elements.displayDelay.value) || 0;
  if (!recentResponse.length) return 0;
  const sorted = [...recentResponse].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function displayDelaySeconds() {
  return delayIsAuto() ? targetDelaySeconds() : Number(elements.displayDelay.value) || 0;
}

// Real-time factor is processing time over the media it covered. Below 1 the
// stream is keeping up and a fixed display delay can hold; above it the backlog
// grows every segment and no fixed offset can track it. The median of the last
// few segments keeps one slow window from swinging the reading.
function recordWork(workSeconds, strideSeconds) {
  if (!(strideSeconds > 0)) return;
  recentWork.push(workSeconds / strideSeconds);
  if (recentWork.length > 5) recentWork.shift();
  const sorted = [...recentWork].sort((a, b) => a - b);
  const rtf = sorted[Math.floor(sorted.length / 2)];
  const keepingUp = rtf < 1;
  elements.rtfBadge.classList.remove("hidden");
  elements.rtfBadge.classList.toggle("warn", !keepingUp);
  elements.rtfValue.textContent = rtf.toFixed(2);
  elements.rtfBadge.title = keepingUp
    ? `Keeping up: ${rtf.toFixed(2)} seconds of work per second of video`
    : `Falling behind: ${rtf.toFixed(2)} seconds of work per second of video`;
}

function resetRtf() {
  recentWork = [];
  elements.rtfBadge.classList.add("hidden");
  elements.rtfBadge.classList.remove("warn");
  elements.rtfValue.textContent = "—";
}

// Rather than deciding an offset before anything is known, playback is nudged
// toward the target: a little slow while it is too close to live, a little fast
// while it has fallen too far behind. The rate stays near 1 so the correction is
// not visible as motion, and a dead band keeps it from hunting.
const RATE_LIMIT = 0.06;
const DEAD_BAND_S = 0.15;
// Rate alone moves the offset by at most RATE_LIMIT seconds per second, so
// closing a multi-second gap that way would take a minute. A large error is
// corrected by seeking once and then held with the rate.
const SEEK_THRESHOLD_S = 0.8;
const SEEK_COOLDOWN_MS = 1000;
// One recorder chunk, so "no delay" still has a frame ready to show.
const LIVE_EDGE_S = 0.3;
let lastSeekAt = 0;

function currentDisplayOffset() {
  if (mode === "file") {
    if (!running) return 0;
    return (performance.now() - streamStartedAt) / 1000 - elements.fileVideo.currentTime;
  }
  const video = elements.cameraDelayed;
  if (!video.buffered.length) return 0;
  return video.buffered.end(video.buffered.length - 1) - video.currentTime;
}

function steerDisplay() {
  const video = mode === "file" ? elements.fileVideo : elements.cameraDelayed;
  const target = targetDelaySeconds();
  // A camera is steered whenever it is being recorded, including before a run,
  // and to the live edge while Auto has no measurement to aim at yet. An
  // uploaded video only moves while it is being analysed.
  const steering = mode === "file" ? running && target > 0 : Boolean(dvr);
  // Aiming at the buffer's exact end leaves the playhead with nothing decoded,
  // so the live edge is held one chunk behind it.
  const aim = mode === "file" ? target : Math.max(target, LIVE_EDGE_S);
  if (!steering || video.paused) {
    if (video.playbackRate !== 1) video.playbackRate = 1;
    return;
  }
  const error = currentDisplayOffset() - aim;
  if (Math.abs(error) <= DEAD_BAND_S) {
    video.playbackRate = 1;
    return;
  }
  const now = performance.now();
  if (Math.abs(error) > SEEK_THRESHOLD_S && now - lastSeekAt > SEEK_COOLDOWN_MS) {
    const limit = mode === "file"
      ? (video.duration || 0)
      : (video.buffered.length ? video.buffered.end(video.buffered.length - 1) : 0);
    const wanted = video.currentTime + error;
    if (limit > 0 && wanted >= 0 && wanted <= limit) {
      video.currentTime = wanted;
      video.playbackRate = 1;
      lastSeekAt = now;
      return;
    }
  }
  // Behind target means the picture is too close to live: slow down to widen it.
  const rate = 1 + Math.max(-RATE_LIMIT, Math.min(RATE_LIMIT, error * 0.15));
  video.playbackRate = rate;
}

function syncDelayBadge() {
  const seconds = displayDelaySeconds();
  const configured = delayIsAuto() || Number(elements.displayDelay.value) > 0;
  elements.delayBadge.classList.toggle("hidden", !configured);
  elements.delayValue.textContent = delayIsAuto()
    ? (seconds > 0 ? `${seconds.toFixed(1)}s auto` : "auto")
    : `${seconds.toFixed(1)}s`;
  elements.displayDelay.disabled = mode === "camera" && !DVR_MIME;
}

// With a delay configured the live picture moves to the inset the moment a run
// starts, so it moves there beforehand too. The arrangement is then the one the
// camera is aimed in, and it is clear before pressing Start that the stage is
// about to carry a delayed picture rather than this one.
function delayConfigured() {
  return mode === "camera" && Boolean(cameraStream) && Boolean(DVR_MIME)
    && (delayIsAuto() || Number(elements.displayDelay.value) > 0);
}

// Recording runs for as long as a delay is selected, not just for as long as a
// run is under way, so the stage carries the delayed picture the run will show
// and the framing can be judged before anything is analysed.
function ensureDvr() {
  if (delayConfigured()) {
    if (!dvr) dvr = startDvr(cameraStream);
  } else if (!running) {
    stopDvr();
  }
  // A refusal to record leaves the live picture on the stage rather than a
  // delayed one that will never arrive.
  return Boolean(dvr);
}

function showSource() {
  const hasFile = mode === "file" && uploaded;
  const hasCamera = mode === "camera" && cameraStream;
  const delayed = ensureDvr();
  elements.fileVideo.classList.toggle("visible", Boolean(hasFile));
  elements.cameraVideo.classList.toggle("visible", Boolean(hasCamera));
  showLiveInset(delayed);
  elements.cameraDelayed.classList.toggle("visible", delayed && dvrHasPicture());
  const covered = hasFile || (hasCamera && (!delayed || dvrHasPicture()));
  elements.emptyStage.classList.toggle("hidden", Boolean(covered));
  elements.stageTitle.textContent = delayed
    ? "The delayed picture appears here"
    : "Choose a video or connect a camera";
  elements.stageNote.textContent = delayed
    ? "LIVE stays in the corner, so the camera can still be aimed."
    : "All media stays on your own hardware.";
}

function dvrHasPicture() {
  return Boolean(dvr) && elements.cameraDelayed.readyState >= 2;
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
function showMeasuredCapture(frames, fps, spanS) {
  if (mode !== "camera" || !running) return;
  const requested = Number(elements.targetFps.value);
  const short = fps < requested * 0.85;
  // A thinned window is only the browser's fault when every frame it sent was
  // used. Once the model falls behind, this server drops the oldest frames to
  // keep the picture live, and blaming the browser for that would send anyone
  // tuning the capture rate after the wrong problem.
  const share = droppedShare();
  const stretched = spanS != null && spanS > 0
    ? ` covering ${spanS.toFixed(1)}s` : "";
  elements.samplingNote.textContent =
    `Capturing ${fps.toFixed(1)} fps of the ${requested} fps requested; `
    + `${frames} frames in this window${stretched}.`
    + (share > 0.01
      ? ` The model is behind, so ${(share * 100).toFixed(0)}% of camera frames`
        + " were dropped to stay live."
      : short ? " The browser cannot keep up at this rate." : "");
  elements.samplingNote.classList.toggle("warn", short || share > 0.01);
}

function noteFrameCounts(message) {
  if (message.received != null) cameraFrameCounts.received = message.received;
  if (message.dropped != null) cameraFrameCounts.dropped = message.dropped;
}

function droppedShare() {
  const { received, dropped } = cameraFrameCounts;
  return received > 0 ? dropped / received : 0;
}

function droppedNote() {
  return `dropping ${(droppedShare() * 100).toFixed(0)}% of frames`;
}

function syncBackendControls() {
  const codec = elements.backend.value === "codec";
  const camera = mode === "camera";
  // Capture rate always decides how many frames a codec window holds: the
  // browser captures at it for the camera, and file segments are cut at it.
  elements.targetFps.disabled = false;
  elements.numFrames.disabled = codec;
  if (codec) {
    const frames = codecWindowFrames();
    const source = camera ? "The browser captures at this rate" : "File segments are cut at this rate";
    elements.samplingNote.textContent = frames < CODEC_MIN_FRAMES
      ? `Codec needs at least ${CODEC_MIN_FRAMES} frames per window; this capture rate and context window give ${frames}. Raise either, or switch to Frames.`
      : `${source}; codec sees ${frames} frames per window and samples them internally. Max frames does not apply.`;
  } else if (camera) {
    elements.samplingNote.textContent = "The browser captures the camera at this rate; each window is capped by Max frames.";
  } else {
    elements.samplingNote.textContent = "Frames backend samples the uploaded video at this rate, capped by Max frames.";
  }
  const tooFew = codec && codecWindowFrames() < CODEC_MIN_FRAMES;
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

function releaseCamera() {
  // The recorder is bound to the stream it was started on, so it goes with it.
  stopDvr();
  if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  elements.cameraVideo.srcObject = null;
}

async function enableCamera() {
  releaseCamera();
  const selected = elements.cameraDevice.value;
  const deviceId = selected;
  cameraStream = await navigator.mediaDevices.getUserMedia({
    video: deviceId ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
                    : { width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  elements.cameraVideo.srcObject = cameraStream;
  await elements.cameraVideo.play();
  const devices = await navigator.mediaDevices.enumerateDevices();
  // Rebuilding the options resets the select, which would silently undo the
  // device that was just chosen and make every switch land on the first camera.
  const chosen = selected || cameraStream.getVideoTracks()[0]?.getSettings().deviceId || "";
  elements.cameraDevice.innerHTML = '<option value="">Default camera</option>';
  devices.filter((device) => device.kind === "videoinput").forEach((device, index) => {
    const option = document.createElement("option");
    option.value = device.deviceId;
    option.textContent = device.label || `Camera ${index + 1}`;
    elements.cameraDevice.append(option);
  });
  elements.cameraDevice.value = chosen;
  syncImmersiveControls();
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
  resetRtf();
  recentResponse = [];
  cameraFrameCounts = { received: 0, dropped: 0 };
  lastSeekAt = 0;
  if (mode === "file") {
    if (!uploaded) { elements.uploadDetail.textContent = "Choose a video first"; return; }
    elements.fileVideo.currentTime = 0;
    // Analysis starts now; playback can start later. Holding the picture back by
    // roughly the time a response takes puts the two in step, so the text lands
    // on the moment it describes instead of trailing it. The badge keeps the
    // offset visible, because the underlying processing is no faster for it.
    socket.send(JSON.stringify({ ...settings("start_file"), media_id: uploaded.id }));
    // Auto has nothing to aim at yet, so it starts live and the corrector widens
    // the gap as soon as the first responses say how wide it should be.
    const delayMs = delayIsAuto() ? 0 : displayDelaySeconds() * 1000;
    if (delayMs > 0) {
      elements.fileVideo.pause();
      delayTimer = setTimeout(() => {
        delayTimer = null;
        if (running) elements.fileVideo.play().catch(() => {});
      }, delayMs);
    } else {
      // Analysis has already been requested, so a refused play must not abort
      // the rest of start(): autoplay policy and backgrounded tabs both reject
      // here, and leaving `running` false would hide a session that is running.
      await elements.fileVideo.play().catch((error) => {
        elements.liveText.textContent = `Playback did not start: ${error.message}`;
      });
    }
  } else {
    if (!cameraStream) await enableCamera();
    socket.send(JSON.stringify(settings("start_camera")));
    const interval = 1000 / Number(elements.targetFps.value);
    captureTimer = setInterval(captureCamera, interval);
    showSource();
  }
  running = true;
  streamStartedAt = performance.now();
  elements.startButton.disabled = true;
  syncImmersiveControls();
  elements.stopButton.disabled = false;
  elements.liveLabel.parentElement.classList.add("running");
  elements.liveLabel.textContent = "LIVE";
}

function stop(notify = true) {
  if (notify && socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ action: "stop" }));
  if (captureTimer) clearInterval(captureTimer);
  captureTimer = null;
  if (delayTimer) clearTimeout(delayTimer);
  delayTimer = null;
  elements.fileVideo.playbackRate = 1;
  elements.fileVideo.pause();
  running = false;
  // A delay left selected keeps recording, so stopping returns to the same
  // preview a run is started from rather than tearing the picture down.
  if (!delayConfigured()) stopDvr();
  showSource();
  elements.startButton.disabled = false;
  syncImmersiveControls();
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
    noteFrameCounts(message);
    if (message.effective_fps) {
      showMeasuredCapture(message.frames, message.effective_fps,
        message.end_s - message.start_s);
    }
    document.querySelector(".video-stage").classList.toggle("processing", message.state === "processing");
    if (message.state === "error") elements.liveText.textContent = message.message;
  } else if (message.type === "token") {
    elements.liveText.textContent = message.text || "…";
    elements.tokenCounter.textContent = `${message.index} TOKENS`;
  } else if (message.type === "result") {
    recordWork(
      (message.prepare_s || 0) + (message.result.metrics.total_s || 0),
      Number(elements.segmentSeconds.value),
    );
    if (message.result.metrics.total_s != null) {
      recentResponse.push((message.backlog_s || 0) + (message.prepare_s || 0)
        + message.result.metrics.total_s);
      if (recentResponse.length > 5) recentResponse.shift();
      if (delayIsAuto()) syncDelayBadge();
    }
    renderResult(message);
  } else if (message.type === "stream" && ["complete", "stopped"].includes(message.state)) {
    stop(false);
  } else if (message.type === "queue") {
    noteFrameCounts(message);
    elements.lagNote.textContent = droppedNote();
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
  noteFrameCounts(message);
  elements.lagMetric.textContent = `${message.lag_s.toFixed(2)}s`;
  // The note has to survive the next result. Reading the drop share here rather
  // than only on the queue message keeps it from flickering back to "behind
  // live edge" between the drops that caused the lag in the first place.
  elements.lagNote.textContent = message.lag_s < 0.1 ? "caught up"
    : droppedShare() > 0.01 ? droppedNote()
    : "behind live edge";
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

new ResizeObserver(syncInsetShape).observe(document.querySelector(".video-stage"));
syncInsetShape();

function updateClock() {
  const seconds = mode === "file" ? elements.fileVideo.currentTime
                                   : (running ? (performance.now() - streamStartedAt) / 1000 : 0);
  elements.timecode.textContent = formatTime(seconds);
  steerDisplay();
  requestAnimationFrame(updateClock);
}

elements.immersiveButton.addEventListener("click", () => setImmersive(!immersiveActive()));
elements.immersiveExit.addEventListener("click", () => setImmersive(false));
elements.immersiveCamera.addEventListener("click", () => { cycleCamera(); });
elements.immersiveToggle.addEventListener("click", () => {
  if (running) stop(); else start();
});
// Leaving fullscreen by gesture or Escape must drop the class too.
document.addEventListener("fullscreenchange", () => {
  if (!document.fullscreenElement && immersiveActive()) setImmersive(false);
});

elements.fileTab.addEventListener("click", () => setMode("file"));
elements.cameraTab.addEventListener("click", () => setMode("camera"));
elements.videoInput.addEventListener("change", (event) => event.target.files[0] && uploadVideo(event.target.files[0]));
elements.cameraDevice.addEventListener("change", () => {
  enableCamera().catch((error) => {
    elements.liveText.textContent = `Could not switch camera: ${error.message}`;
  });
});
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
elements.displayDelay.addEventListener("change", () => { syncDelayBadge(); showSource(); });
syncDelayBadge();
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
