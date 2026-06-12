import { initDetector, detectPersons } from './detect';
import { buildCropPath, cropAt, Tracker, type Point, type Sample, type CropKey } from './path';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const fileInput = $<HTMLInputElement>('fileInput');
const analyzeBtn = $<HTMLButtonElement>('analyzeBtn');
const playBtn = $<HTMLButtonElement>('playBtn');
const strideSel = $<HTMLSelectElement>('strideSel');
const padSlider = $<HTMLInputElement>('padSlider');
const padValue = $<HTMLSpanElement>('padValue');
const smoothSlider = $<HTMLInputElement>('smoothSlider');
const smoothValue = $<HTMLSpanElement>('smoothValue');
const showBoxes = $<HTMLInputElement>('showBoxes');
const statusEl = $<HTMLParagraphElement>('status');
const progressBar = $<HTMLProgressElement>('progressBar');
const video = $<HTMLVideoElement>('video');
const originalCanvas = $<HTMLCanvasElement>('originalCanvas');
const previewCanvas = $<HTMLCanvasElement>('previewCanvas');
const originalCtx = originalCanvas.getContext('2d')!;
const previewCtx = previewCanvas.getContext('2d')!;

const MAX_ZOOM = 4;
const PREVIEW_MAX_W = 1280;
const ORIGINAL_MAX_W = 640;

let samples: Sample[] = [];
let cropPath: CropKey[] = [];
let analyzing = false;
let rafHandle = 0;
let seedPoint: Point | null = null;

function setStatus(msg: string, isError = false) {
  statusEl.textContent = msg;
  statusEl.classList.toggle('error', isError);
}

function pathOptions() {
  return {
    padFactor: parseFloat(padSlider.value),
    smoothSigmaSec: parseFloat(smoothSlider.value),
    maxZoom: MAX_ZOOM,
    sampleInterval: parseFloat(strideSel.value),
  };
}

function rebuildPath() {
  if (samples.length === 0) return;
  cropPath = buildCropPath(samples, video.videoWidth, video.videoHeight, pathOptions());
  drawFrame();
}

function sizeCanvases() {
  const { videoWidth: w, videoHeight: h } = video;
  const oScale = Math.min(1, ORIGINAL_MAX_W / w);
  originalCanvas.width = Math.round(w * oScale);
  originalCanvas.height = Math.round(h * oScale);
  const pScale = Math.min(1, PREVIEW_MAX_W / w);
  previewCanvas.width = Math.round(w * pScale);
  previewCanvas.height = Math.round(h * pScale);
}

function nearestSample(t: number): Sample | undefined {
  if (samples.length === 0) return undefined;
  let best = samples[0];
  for (const s of samples) {
    if (Math.abs(s.t - t) < Math.abs(best.t - t)) best = s;
  }
  return best;
}

function drawFrame() {
  if (!video.videoWidth) return;
  const t = video.currentTime;
  const ow = originalCanvas.width;
  const oh = originalCanvas.height;
  const scale = ow / video.videoWidth;

  originalCtx.drawImage(video, 0, 0, ow, oh);

  const crop = cropPath.length ? cropAt(cropPath, t) : null;

  if (showBoxes.checked) {
    const box = nearestSample(t)?.box;
    if (box) {
      originalCtx.strokeStyle = '#ff5c5c';
      originalCtx.lineWidth = 2;
      originalCtx.strokeRect(
        (box.cx - box.w / 2) * scale,
        (box.cy - box.h / 2) * scale,
        box.w * scale,
        box.h * scale,
      );
    }
    if (crop) {
      originalCtx.strokeStyle = '#35c4a2';
      originalCtx.lineWidth = 2;
      originalCtx.strokeRect(
        (crop.cx - crop.cropW / 2) * scale,
        (crop.cy - crop.cropH / 2) * scale,
        crop.cropW * scale,
        crop.cropH * scale,
      );
    }
  }

  if (seedPoint && samples.length === 0) {
    const sx = seedPoint.x * scale;
    const sy = seedPoint.y * scale;
    originalCtx.strokeStyle = '#ffd24a';
    originalCtx.lineWidth = 2;
    originalCtx.beginPath();
    originalCtx.arc(sx, sy, 14, 0, Math.PI * 2);
    originalCtx.moveTo(sx - 20, sy);
    originalCtx.lineTo(sx + 20, sy);
    originalCtx.moveTo(sx, sy - 20);
    originalCtx.lineTo(sx, sy + 20);
    originalCtx.stroke();
  }

  if (crop) {
    previewCtx.drawImage(
      video,
      crop.cx - crop.cropW / 2,
      crop.cy - crop.cropH / 2,
      crop.cropW,
      crop.cropH,
      0,
      0,
      previewCanvas.width,
      previewCanvas.height,
    );
  } else {
    previewCtx.drawImage(video, 0, 0, previewCanvas.width, previewCanvas.height);
  }
}

function renderLoop() {
  drawFrame();
  rafHandle = requestAnimationFrame(renderLoop);
}

function seekTo(t: number): Promise<void> {
  return new Promise((resolve) => {
    video.addEventListener('seeked', () => resolve(), { once: true });
    video.currentTime = t;
  });
}

async function analyze() {
  if (!seedPoint) {
    setStatus('Click on yourself in the left frame first, so I know who to track.', true);
    return;
  }
  analyzing = true;
  analyzeBtn.disabled = true;
  playBtn.disabled = true;
  video.pause();
  samples = [];
  cropPath = [];

  const { videoWidth: w, videoHeight: h } = video;
  const tracker = new Tracker(seedPoint, w, h);
  const interval = parseFloat(strideSel.value);
  const duration = video.duration;
  const total = Math.floor(duration / interval) + 1;
  progressBar.hidden = false;
  const startedAt = performance.now();

  let detected = 0;
  let roiHits = 0;
  for (let i = 0; i * interval < duration; i++) {
    const t = i * interval;
    await seekTo(t);
    let box = null;
    try {
      // Pass 1: full frame. Pass 2: zoomed window around the predicted
      // position — a rider that's 20px tall in a downscaled 4K frame is
      // often only findable this way.
      box = tracker.match(await detectPersons(video, w, h), t);
      if (!box) {
        const roi = tracker.searchRegion(t);
        box = tracker.match(await detectPersons(video, w, h, roi), t);
        if (box) roiHits++;
      }
    } catch (e) {
      setStatus(`Detection failed at ${t.toFixed(1)}s: ${e}`, true);
      break;
    }
    if (box) detected++;
    samples.push({ t, box });
    progressBar.value = (i + 1) / total;
    if (i % 5 === 0) {
      setStatus(
        `Analyzing… ${t.toFixed(1)}s / ${duration.toFixed(1)}s — rider in ${detected}/${i + 1} samples (${roiHits} via zoomed re-detect)`,
      );
      drawFrame();
    }
  }

  const secs = ((performance.now() - startedAt) / 1000).toFixed(1);
  progressBar.hidden = true;
  analyzing = false;
  analyzeBtn.disabled = false;
  playBtn.disabled = false;

  rebuildPath();
  await seekTo(0);
  drawFrame();
  setStatus(
    `Done in ${secs}s — rider detected in ${detected}/${samples.length} samples. ` +
      `Tune the sliders (instant, no re-analysis needed) and hit Play.`,
  );
}

fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  video.src = URL.createObjectURL(file);
  samples = [];
  cropPath = [];
  seedPoint = null;
  video.addEventListener(
    'loadedmetadata',
    () => {
      sizeCanvases();
      void seekTo(0.01).then(drawFrame);
      analyzeBtn.disabled = false;
      playBtn.disabled = false;
      setStatus(
        `${file.name} — ${video.videoWidth}×${video.videoHeight}, ${video.duration.toFixed(1)}s. ` +
          `Click on yourself in the left frame, then Analyze.`,
      );
    },
    { once: true },
  );
});

originalCanvas.addEventListener('click', (e) => {
  if (analyzing || !video.videoWidth) return;
  const rect = originalCanvas.getBoundingClientRect();
  seedPoint = {
    x: ((e.clientX - rect.left) / rect.width) * video.videoWidth,
    y: ((e.clientY - rect.top) / rect.height) * video.videoHeight,
  };
  samples = [];
  cropPath = [];
  drawFrame();
  setStatus('Target set. Hit Analyze. (Scrub-free: tracking always starts from the first frame.)');
});

analyzeBtn.addEventListener('click', () => void analyze());

playBtn.addEventListener('click', () => {
  if (video.paused) {
    void video.play();
    playBtn.textContent = 'Pause';
  } else {
    video.pause();
    playBtn.textContent = 'Play';
  }
});

video.addEventListener('ended', () => {
  playBtn.textContent = 'Play';
});

padSlider.addEventListener('input', () => {
  padValue.textContent = `${parseFloat(padSlider.value).toFixed(1)}×`;
  if (!analyzing) rebuildPath();
});
smoothSlider.addEventListener('input', () => {
  smoothValue.textContent = `${parseFloat(smoothSlider.value).toFixed(1)}s`;
  if (!analyzing) rebuildPath();
});
showBoxes.addEventListener('change', drawFrame);

initDetector()
  .then((ep) => setStatus(`Detector ready (${ep === 'webgpu' ? 'WebGPU 🚀' : 'WASM/CPU — slower'}). Pick a clip.`))
  .catch((e) => setStatus(`${e}`, true));

renderLoop();
window.addEventListener('beforeunload', () => cancelAnimationFrame(rafHandle));
