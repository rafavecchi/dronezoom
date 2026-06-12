import { initDetector, detectPersons } from './detect';
import { exportVideo } from './export';
import { buildCropPath, cropAt, gaussianSmooth, type Sample, type CropKey } from './path';
import { estimateCameraPath, lerpSeries, stepSeries, type CameraPath } from './stabilize';
import { Tracker, type Point } from './tracker';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const fileInput = $<HTMLInputElement>('fileInput');
const analyzeBtn = $<HTMLButtonElement>('analyzeBtn');
const playBtn = $<HTMLButtonElement>('playBtn');
const exportBtn = $<HTMLButtonElement>('exportBtn');
const strideSel = $<HTMLSelectElement>('strideSel');
const padSlider = $<HTMLInputElement>('padSlider');
const padValue = $<HTMLSpanElement>('padValue');
const smoothSlider = $<HTMLInputElement>('smoothSlider');
const smoothValue = $<HTMLSpanElement>('smoothValue');
const showBoxes = $<HTMLInputElement>('showBoxes');
const stabilizeChk = $<HTMLInputElement>('stabilizeChk');
const statusEl = $<HTMLParagraphElement>('status');
const scrubBar = $<HTMLInputElement>('scrubBar');
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
let seedT = 0;
let seedDirty = false;
let scrubbing = false;
let camPath: CameraPath | null = null;
// mediaTime of the actually-presented video frame. video.currentTime is
// the playback head and can lead the displayed frame by up to a frame —
// applying a per-frame shake correction at the wrong frame *adds* jitter
// instead of canceling it.
let displayT: number | null = null;
// Smoothed roll: the intended slow camera leveling; the residual
// (instantaneous minus smoothed) is the roll shake we counter-rotate.
let smoothRoll: number[] = [];
// Smoothed translation: used to express frame-boundary limits in world
// coordinates without injecting per-frame jitter into the path.
let smoothCamX: number[] = [];
let smoothCamY: number[] = [];

function clampNum(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

interface Cam {
  x: number;
  y: number;
  r: number;
}

function camOffset(t: number): Cam {
  if (!stabilizeChk.checked || !camPath) return { x: 0, y: 0, r: 0 };
  // Step lookup: a correction belongs to exactly one frame.
  return {
    x: stepSeries(camPath.ts, camPath.xs, t),
    y: stepSeries(camPath.ts, camPath.ys, t),
    r: stepSeries(camPath.ts, camPath.rs, t),
  };
}

function smoothRollAt(t: number): number {
  return stabilizeChk.checked && camPath && smoothRoll.length
    ? lerpSeries(camPath.ts, smoothRoll, t)
    : 0;
}

// Frame content = world rotated by cam.r about the frame center, then
// translated by (cam.x, cam.y).
function worldToFrame(x: number, y: number, cam: Cam): { x: number; y: number } {
  const fcx = video.videoWidth / 2;
  const fcy = video.videoHeight / 2;
  const cos = Math.cos(cam.r);
  const sin = Math.sin(cam.r);
  const dx = x - fcx;
  const dy = y - fcy;
  return { x: fcx + cam.x + dx * cos - dy * sin, y: fcy + cam.y + dx * sin + dy * cos };
}

function frameToWorld(x: number, y: number, cam: Cam): { x: number; y: number } {
  const fcx = video.videoWidth / 2;
  const fcy = video.videoHeight / 2;
  const cos = Math.cos(-cam.r);
  const sin = Math.sin(-cam.r);
  const dx = x - fcx - cam.x;
  const dy = y - fcy - cam.y;
  return { x: fcx + dx * cos - dy * sin, y: fcy + dx * sin + dy * cos };
}

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
  const { videoWidth: w, videoHeight: h } = video;
  if (camPath) {
    // Per-frame samples are ~uniformly spaced, so sigma in samples is
    // smoothing seconds divided by the average frame interval.
    const dtAvg =
      camPath.ts.length > 1 ? camPath.ts[camPath.ts.length - 1] / (camPath.ts.length - 1) : 1;
    const sigma = parseFloat(smoothSlider.value) / dtAvg;
    smoothRoll = gaussianSmooth(camPath.rs, sigma);
    smoothCamX = gaussianSmooth(camPath.xs, sigma);
    smoothCamY = gaussianSmooth(camPath.ys, sigma);
  }
  // Move detections into stabilized "world" coordinates before smoothing:
  // shake is removed from the subject signal, and the renderer applies
  // each frame's camera transform back so it cancels exactly.
  const worldSamples = samples.map((s) => {
    if (!s.box) return s;
    const wpt = frameToWorld(s.box.cx, s.box.cy, camOffset(s.t));
    return { t: s.t, box: { ...s.box, cx: wpt.x, cy: wpt.y } };
  });
  const useCam = stabilizeChk.checked && camPath !== null;
  // Margin reserved so the renderer's per-frame shake correction (the
  // difference between instantaneous and smoothed camera) has headroom
  // before hitting the hard frame-boundary clamp.
  const mX = 0.02 * w;
  const mY = 0.025 * h;
  cropPath = buildCropPath(worldSamples, w, h, {
    ...pathOptions(),
    constrain: (t, cropW, cropH, x, y) => {
      const cam: Cam = useCam
        ? {
            x: lerpSeries(camPath!.ts, smoothCamX, t),
            y: lerpSeries(camPath!.ts, smoothCamY, t),
            r: lerpSeries(camPath!.ts, smoothRoll, t),
          }
        : { x: 0, y: 0, r: 0 };
      const f = worldToFrame(x, y, cam);
      const loX = cropW / 2 + mX;
      const hiX = w - cropW / 2 - mX;
      const loY = cropH / 2 + mY;
      const hiY = h - cropH / 2 - mY;
      const fx = hiX < loX ? w / 2 : clampNum(f.x, loX, hiX);
      const fy = hiY < loY ? h / 2 : clampNum(f.y, loY, hiY);
      if (fx === f.x && fy === f.y) return { x, y };
      return frameToWorld(fx, fy, cam);
    },
  });
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

interface CropState {
  crop: CropKey;
  x: number;
  y: number;
  roll: number;
}

function cropStateAt(t: number): CropState | null {
  if (!cropPath.length) return null;
  const crop = cropAt(cropPath, t);
  const cam = camOffset(t);
  const q = worldToFrame(crop.cx, crop.cy, cam);
  return {
    crop,
    x: clampNum(q.x, crop.cropW / 2, video.videoWidth - crop.cropW / 2),
    y: clampNum(q.y, crop.cropH / 2, video.videoHeight - crop.cropH / 2),
    roll: cam.r - smoothRollAt(t),
  };
}

// The single source of truth for the output image — used by both the
// live preview and the exporter.
function renderView(c2: CanvasRenderingContext2D, outW: number, outH: number, t: number) {
  const st = cropStateAt(t);
  if (!st) {
    c2.drawImage(video, 0, 0, outW, outH);
    return;
  }
  // Sample the source rotated by the inverse roll residual about the
  // crop center, so roll shake cancels in the output.
  const k = outW / st.crop.cropW;
  c2.setTransform(1, 0, 0, 1, 0, 0);
  c2.translate(outW / 2, outH / 2);
  c2.scale(k, k);
  c2.rotate(-st.roll);
  c2.translate(-st.x, -st.y);
  c2.drawImage(video, 0, 0);
  c2.setTransform(1, 0, 0, 1, 0, 0);
}

function drawFrame() {
  if (!video.videoWidth) return;
  const t = displayT ?? video.currentTime;
  const ow = originalCanvas.width;
  const oh = originalCanvas.height;
  const scale = ow / video.videoWidth;

  originalCtx.drawImage(video, 0, 0, ow, oh);

  const st = cropStateAt(t);

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
    if (st) {
      originalCtx.strokeStyle = '#35c4a2';
      originalCtx.lineWidth = 2;
      originalCtx.strokeRect(
        (st.x - st.crop.cropW / 2) * scale,
        (st.y - st.crop.cropH / 2) * scale,
        st.crop.cropW * scale,
        st.crop.cropH * scale,
      );
    }
  }

  if (seedPoint && (seedDirty || samples.length === 0)) {
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

  renderView(previewCtx, previewCanvas.width, previewCanvas.height, t);
}

function renderLoop() {
  drawFrame();
  if (!scrubbing && !analyzing && video.videoWidth) {
    scrubBar.value = String(video.currentTime);
  }
  rafHandle = requestAnimationFrame(renderLoop);
}

function seekTo(t: number): Promise<void> {
  return new Promise((resolve) => {
    video.addEventListener('seeked', () => resolve(), { once: true });
    video.currentTime = t;
  });
}

// Track the mediaTime of every presented frame. Registered once; the
// callback chain survives src changes.
let presentLoopStarted = false;
function startPresentLoop() {
  if (presentLoopStarted) return;
  const rvfc = (
    video as unknown as {
      requestVideoFrameCallback?: (cb: (now: number, meta: { mediaTime: number }) => void) => number;
    }
  ).requestVideoFrameCallback?.bind(video);
  if (!rvfc) return; // drawFrame falls back to video.currentTime
  presentLoopStarted = true;
  const cb = (_now: number, meta: { mediaTime: number }) => {
    displayT = meta.mediaTime;
    rvfc(cb);
  };
  rvfc(cb);
}
video.addEventListener('seeked', () => {
  displayT = video.currentTime;
});

async function analyze() {
  if (!seedPoint) {
    setStatus('Click on yourself in the left frame first, so I know who to track.', true);
    return;
  }
  analyzing = true;
  seedDirty = false;
  analyzeBtn.disabled = true;
  playBtn.disabled = true;
  video.pause();
  samples = [];
  cropPath = [];

  const { videoWidth: w, videoHeight: h } = video;
  const interval = parseFloat(strideSel.value);
  const duration = video.duration;
  const seedTime = Math.min(seedT, Math.max(0, duration - 0.05));
  const total = Math.floor(duration / interval) + 1;
  progressBar.hidden = false;
  const startedAt = performance.now();

  let detected = 0;
  let roiHits = 0;
  let processed = 0;

  // Full-frame pass first; if that misses, a zoomed pass around the
  // predicted position — a rider ~20px tall in downscaled 4K is often
  // only findable the second way.
  const detectAt = async (tracker: Tracker, t: number) => {
    await seekTo(t);
    let box = tracker.match(video, await detectPersons(video, w, h), t);
    if (!box) {
      box = tracker.match(video, await detectPersons(video, w, h, tracker.searchRegion(t)), t);
      if (box) roiHits++;
    }
    processed++;
    if (box) detected++;
    progressBar.value = processed / total;
    if (processed % 5 === 0) {
      setStatus(
        `Analyzing… ${processed}/${total} samples — rider in ${detected} (${roiHits} via zoomed re-detect)`,
      );
      drawFrame();
    }
    return box;
  };

  try {
    // Track outward from the seeded frame in both directions, sharing the
    // appearance template, so the click anchors identity for the whole clip.
    const forward = new Tracker({ ...seedPoint }, w, h);
    const fwdSamples: Sample[] = [];
    for (let t = seedTime; t < duration; t += interval) {
      fwdSamples.push({ t, box: await detectAt(forward, t) });
    }
    const backward = new Tracker({ ...seedPoint }, w, h, forward.appearance);
    const bwdSamples: Sample[] = [];
    for (let t = seedTime - interval; t >= 0; t -= interval) {
      bwdSamples.push({ t, box: await detectAt(backward, t) });
    }
    samples = [...bwdSamples.reverse(), ...fwdSamples];
  } catch (e) {
    setStatus(`Detection failed: ${e}`, true);
    samples = [];
  }

  if (samples.length > 0) {
    setStatus('Measuring per-frame camera shake (plays the clip through once)…');
    progressBar.value = 0;
    // Rider position over time (frame coords), so the stabilizer can
    // measure motion in the background around the rider instead of the
    // parallax-ambiguous full frame.
    const riderTs: number[] = [];
    const riderXs: number[] = [];
    const riderYs: number[] = [];
    for (const s of samples) {
      if (s.box) {
        riderTs.push(s.t);
        riderXs.push(s.box.cx);
        riderYs.push(s.box.cy);
      }
    }
    const riderAt = (t: number) =>
      riderTs.length > 0
        ? { x: lerpSeries(riderTs, riderXs, t), y: lerpSeries(riderTs, riderYs, t) }
        : { x: w / 2, y: h / 2 };
    try {
      camPath = await estimateCameraPath(video, riderAt, (t) => {
        progressBar.value = t / duration;
      });
    } catch (e) {
      camPath = null;
      setStatus(`Shake measurement unavailable (${e}) — continuing without stabilization.`, true);
    }
  }

  const secs = ((performance.now() - startedAt) / 1000).toFixed(1);
  progressBar.hidden = true;
  analyzing = false;
  analyzeBtn.disabled = false;
  playBtn.disabled = false;
  if (samples.length === 0) return;
  exportBtn.disabled = false;

  rebuildPath();
  await seekTo(0);
  drawFrame();
  const covNote = camPath
    ? `, shake measured for ${(camPath.coverage * 100).toFixed(0)}% of frames` +
      (camPath.coverage < 0.85 ? ' (low — uncovered frames stay shaky)' : '')
    : '';
  setStatus(
    `Done in ${secs}s — rider in ${detected}/${samples.length} samples (${roiHits} via zoomed re-detect)${covNote}. ` +
      `Toggle "stabilize" to compare; if tracking drifts, click yourself at that moment and re-Analyze.`,
  );
}

fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  video.src = URL.createObjectURL(file);
  samples = [];
  cropPath = [];
  camPath = null;
  seedPoint = null;
  seedDirty = false;
  displayT = null;
  exportBtn.disabled = true;
  video.addEventListener(
    'loadedmetadata',
    () => {
      startPresentLoop();
      sizeCanvases();
      void seekTo(0.01).then(drawFrame);
      analyzeBtn.disabled = false;
      playBtn.disabled = false;
      scrubBar.disabled = false;
      scrubBar.max = String(video.duration);
      scrubBar.value = '0';
      setStatus(
        `${file.name} — ${video.videoWidth}×${video.videoHeight}, ${video.duration.toFixed(1)}s. ` +
          `Find yourself (scrub to any frame) and click on yourself in the left view, then Analyze.`,
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
  seedT = video.currentTime;
  seedDirty = true;
  drawFrame();
  setStatus(
    `Target set at ${seedT.toFixed(1)}s — hit Analyze. Tracking runs forward and backward from this frame.`,
  );
});

scrubBar.addEventListener('input', () => {
  if (analyzing) return;
  video.pause();
  playBtn.textContent = 'Play';
  video.currentTime = parseFloat(scrubBar.value);
});
scrubBar.addEventListener('pointerdown', () => (scrubbing = true));
window.addEventListener('pointerup', () => (scrubbing = false));

function estimateFps(): number {
  if (!camPath || camPath.ts.length < 10) return 30;
  let minGap = Infinity;
  for (let i = 1; i < camPath.ts.length; i++) {
    const gap = camPath.ts[i] - camPath.ts[i - 1];
    if (gap > 1e-4) minGap = Math.min(minGap, gap);
  }
  const raw = 1 / minGap;
  const common = [23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60];
  let best = 30;
  for (const f of common) if (Math.abs(f - raw) < Math.abs(best - raw)) best = f;
  return best;
}

exportBtn.addEventListener('click', async () => {
  if (analyzing || cropPath.length === 0) return;
  analyzing = true; // blocks slider rebuilds and scrub during export
  analyzeBtn.disabled = true;
  playBtn.disabled = true;
  exportBtn.disabled = true;
  video.pause();
  playBtn.textContent = 'Play';
  progressBar.hidden = false;
  progressBar.value = 0;

  const fps = estimateFps();
  const aspect = video.videoWidth / video.videoHeight;
  const outW = Math.min(1920, video.videoWidth);
  const outH = Math.round(outW / aspect / 2) * 2;
  const startedAt = performance.now();

  try {
    const blob = await exportVideo({
      video,
      fps,
      outW,
      outH,
      render: renderView,
      seekTo,
      onProgress: (frac) => {
        progressBar.value = frac;
        if (Math.round(frac * 100) % 5 === 0) {
          setStatus(`Exporting… ${(frac * 100).toFixed(0)}% (${outW}×${outH} @ ${fps}fps)`);
        }
      },
    });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'dronezoom-export.mp4';
    a.click();
    const secs = ((performance.now() - startedAt) / 1000).toFixed(0);
    setStatus(
      `Exported ${(blob.size / 1e6).toFixed(1)} MB in ${secs}s (${outW}×${outH} @ ${fps}fps) — check your downloads.`,
    );
  } catch (e) {
    setStatus(`Export failed: ${e}`, true);
  } finally {
    analyzing = false;
    analyzeBtn.disabled = false;
    playBtn.disabled = false;
    exportBtn.disabled = false;
    progressBar.hidden = true;
  }
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
stabilizeChk.addEventListener('change', () => {
  if (!analyzing) rebuildPath();
});

initDetector()
  .then((ep) => setStatus(`Detector ready (${ep === 'webgpu' ? 'WebGPU 🚀' : 'WASM/CPU — slower'}). Pick a clip.`))
  .catch((e) => setStatus(`${e}`, true));

renderLoop();
window.addEventListener('beforeunload', () => cancelAnimationFrame(rafHandle));
