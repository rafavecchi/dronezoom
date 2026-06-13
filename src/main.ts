import { runAnalysis, probeFps, type AnalysisResult } from './analyze';
import { initDetector } from './detect';
import { getClientId, persistExport, restoreExport, setClientId, uploadToDrive } from './drive';
import { exportVideo } from './export';
import { apply, buildResiduals, decompose, IDENTITY, invert, type Affine } from './motion';
import { buildCropPath, cropAt, gaussianSmooth, type Sample, type CropKey } from './path';
import type { Point } from './tracker';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const fileInput = $<HTMLInputElement>('fileInput');
const analyzeBtn = $<HTMLButtonElement>('analyzeBtn');
const playBtn = $<HTMLButtonElement>('playBtn');
const exportBtn = $<HTMLButtonElement>('exportBtn');
const driveBtn = $<HTMLButtonElement>('driveBtn');
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
const SAMPLE_DT = 0.2;

let analysis: AnalysisResult | null = null;
let Ds: Affine[] = [];
let samples: Sample[] = []; // YOLO samples, overlay only
let cropPath: CropKey[] = [];
let analyzing = false;
let rafHandle = 0;
let seedPoint: Point | null = null;
let seedT = 0;
let seedDirty = false;
let scrubbing = false;
// mediaTime of the actually-presented frame; video.currentTime can lead
// the displayed frame, which would misapply per-frame corrections.
let displayT: number | null = null;
let lastExport: { blob: Blob; name: string } | null = null;
let clipName = 'clip';

function clampNum(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
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
    sampleInterval: SAMPLE_DT,
  };
}

function frameIndexAt(t: number): number {
  if (!analysis) return 0;
  return clampNum(Math.floor(t * analysis.fps + 1e-3), 0, analysis.frames - 1);
}

function residualAt(t: number): Affine {
  if (!stabilizeChk.checked || Ds.length === 0) return IDENTITY;
  return Ds[clampNum(frameIndexAt(t), 0, Ds.length - 1)];
}

function rebuildPath() {
  if (!analysis) return;
  const { videoWidth: w, videoHeight: h } = video;
  const smoothSec = parseFloat(smoothSlider.value);
  const fps = analysis.fps;
  const hSmooth = gaussianSmooth(Array.from(analysis.blobH), fps * 0.5);

  // Zoom-out-on-pan: widen the crop while the camera moves violently —
  // "go wide when the action gets fast". Envelope (rolling max) keeps
  // whip magnitude; thresholds scale with resolution (tuned at 1080p).
  const speeds = analysis.incs.map((inc) => Math.hypot(inc.dtx, inc.dty));
  const r = Math.round(fps * 0.3);
  const env = speeds.map((_, i) =>
    Math.max(...speeds.slice(Math.max(0, i - r), Math.min(speeds.length, i + r + 1))),
  );
  const spSmooth = gaussianSmooth(env, fps * 0.3);
  const lo = 0.021 * w;
  const hi = 0.052 * w;
  const widenArr = spSmooth.map((s) => {
    const x01 = clampNum((s - lo) / (hi - lo), 0, 1);
    return 1 + 0.8 * x01 * x01 * (3 - 2 * x01);
  });
  const widen = (t: number) =>
    widenArr[clampNum(Math.floor(t * fps), 0, widenArr.length - 1)] ?? 1;

  // Shake residuals, capped to what the crop's leash can absorb: full
  // NOTE: the correction-cap (attenuateResiduals) was a smoothness
  // regression (user-confirmed) — reverted to full shake cancellation.
  Ds = buildResiduals(analysis.incs, fps, smoothSec);

  // Rider track -> "intended camera" coordinates (shake removed), then
  // downsample to the path grid.
  const trackSamples: Sample[] = [];
  for (let t = 0; t < analysis.frames / fps; t += SAMPLE_DT) {
    const k = frameIndexAt(t);
    const Di = invert(Ds[clampNum(k, 0, Ds.length - 1)]);
    const p = apply(Di, analysis.blobX[k], analysis.blobY[k]);
    trackSamples.push({
      t,
      box: {
        cx: p.x,
        cy: p.y,
        w: hSmooth[k] * 0.8,
        h: clampNum(hSmooth[k] * 1.6, 0.04 * h, 0.3 * h),
        score: 1,
      },
    });
  }

  const mX = 0.02 * w;
  const mY = 0.025 * h;
  cropPath = buildCropPath(trackSamples, w, h, {
    ...pathOptions(),
    widen,
    constrain: (_t, cropW, cropH, x, y) => {
      const loX = cropW / 2 + mX;
      const hiX = w - cropW / 2 - mX;
      const loY = cropH / 2 + mY;
      const hiY = h - cropH / 2 - mY;
      return {
        x: hiX < loX ? w / 2 : clampNum(x, loX, hiX),
        y: hiY < loY ? h / 2 : clampNum(y, loY, hiY),
      };
    },
  });

  (window as unknown as Record<string, unknown>).__dz = {
    analysis,
    Ds,
    get cropPath() {
      return cropPath;
    },
    opts: pathOptions(),
  };
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
  theta: number;
  scale: number;
}

function cropStateAt(t: number): CropState | null {
  if (!cropPath.length) return null;
  const crop = cropAt(cropPath, t);
  const D = residualAt(t);
  const q = apply(D, crop.cx, crop.cy);
  const dec = decompose(D);
  return {
    crop,
    x: clampNum(q.x, crop.cropW / 2, video.videoWidth - crop.cropW / 2),
    y: clampNum(q.y, crop.cropH / 2, video.videoHeight - crop.cropH / 2),
    theta: dec.theta,
    scale: dec.scale,
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
  // Counter-rotate/scale by the residual about the crop center so the
  // shake cancels in the output.
  const zoom = outW / (st.crop.cropW * st.scale);
  c2.setTransform(1, 0, 0, 1, 0, 0);
  c2.translate(outW / 2, outH / 2);
  c2.scale(zoom, zoom);
  c2.rotate(-st.theta);
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
    if (analysis) {
      const k = frameIndexAt(t);
      originalCtx.strokeStyle = '#ffd24a';
      originalCtx.lineWidth = 2;
      originalCtx.beginPath();
      originalCtx.arc(analysis.blobX[k] * scale, analysis.blobY[k] * scale, 10, 0, Math.PI * 2);
      originalCtx.stroke();
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

  if (seedPoint && (seedDirty || !analysis)) {
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
    const wait = document.hidden
      ? new Promise<void>((r) => {
          const h = () => {
            if (!document.hidden) {
              document.removeEventListener('visibilitychange', h);
              r();
            }
          };
          document.addEventListener('visibilitychange', h);
        })
      : Promise.resolve();
    void wait.then(() => {
      let settled = false;
      const finish = () => {
        if (settled) return;
        settled = true;
        video.removeEventListener('seeked', finish);
        clearInterval(nudge);
        resolve();
      };
      video.addEventListener('seeked', finish);
      const nudge = setInterval(() => {
        if (document.hidden) return;
        if (Math.abs(video.currentTime - t) < 0.02 && video.readyState >= 2) finish();
        else video.currentTime = t;
      }, 3000);
      video.currentTime = t;
    });
  });
}

// Keep the screen awake during long passes — a locked phone freezes the
// page and stalls analysis/export.
let wakeLock: { release(): Promise<void> } | null = null;
async function acquireWakeLock() {
  try {
    const wl = (
      navigator as unknown as {
        wakeLock?: { request(type: 'screen'): Promise<{ release(): Promise<void> }> };
      }
    ).wakeLock;
    if (wl) wakeLock = await wl.request('screen');
  } catch {
    // denied/unsupported — analysis still works, screen may sleep
  }
}
function releaseWakeLock() {
  void wakeLock?.release().catch(() => undefined);
  wakeLock = null;
}
document.addEventListener('visibilitychange', () => {
  // wake locks auto-release when the tab hides; re-grab on return
  if (!document.hidden && analyzing) void acquireWakeLock();
});

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
  exportBtn.disabled = true;
  video.pause();
  playBtn.textContent = 'Play';
  analysis = null;
  cropPath = [];
  samples = [];
  progressBar.hidden = false;
  progressBar.value = 0;
  const startedAt = performance.now();
  await acquireWakeLock();

  try {
    setStatus('Probing frame rate…');
    const fps = await probeFps(video);
    setStatus(
      `Analyzing every frame @ ${fps}fps — screen stays awake; if you switch apps it pauses and resumes when you return.`,
    );
    let lastDraw = 0;
    analysis = await runAnalysis(video, seedPoint, seedT, fps, (done, total, roi, blob) => {
      progressBar.value = done / total;
      if (performance.now() - lastDraw > 500) {
        lastDraw = performance.now();
        setStatus(
          `Analyzing… frame ${done}/${total} — rider blob on ${blob}, YOLO anchors ${roi}`,
        );
        drawFrame();
      }
    });
    samples = analysis.samples;
  } catch (e) {
    setStatus(`Analysis failed: ${e}`, true);
    analysis = null;
  }

  const secs = ((performance.now() - startedAt) / 1000).toFixed(0);
  progressBar.hidden = true;
  analyzing = false;
  releaseWakeLock();
  analyzeBtn.disabled = false;
  playBtn.disabled = false;
  if (!analysis) return;
  exportBtn.disabled = false;

  rebuildPath();
  await seekTo(0);
  drawFrame();
  const blobPct = (
    (Array.from(analysis.blobFound).reduce((s, v) => s + v, 0) / analysis.frames) *
    100
  ).toFixed(0);
  setStatus(
    `Done in ${secs}s — rider blob tracked on ${blobPct}% of frames. ` +
      `Scrub through it; if the yellow circle drifts off you, click yourself there and re-Analyze.`,
  );
}

fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  clipName = file.name.replace(/\.[^.]+$/, '');
  // keep lastExport: → Drive can still upload the previous export
  video.src = URL.createObjectURL(file);
  analysis = null;
  samples = [];
  cropPath = [];
  Ds = [];
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

exportBtn.addEventListener('click', async () => {
  if (analyzing || cropPath.length === 0 || !analysis) return;
  analyzing = true; // blocks slider rebuilds and scrubbing during export
  analyzeBtn.disabled = true;
  playBtn.disabled = true;
  exportBtn.disabled = true;
  video.pause();
  playBtn.textContent = 'Play';
  progressBar.hidden = false;
  progressBar.value = 0;
  await acquireWakeLock();

  const fps = analysis.fps;
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
    const name = `dronezoom-${clipName}.mp4`;
    lastExport = { blob, name };
    driveBtn.disabled = false;
    void persistExport(blob, name);
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    const secs = ((performance.now() - startedAt) / 1000).toFixed(0);
    setStatus(
      `Exported ${(blob.size / 1e6).toFixed(1)} MB in ${secs}s (${outW}×${outH} @ ${fps}fps) — ` +
        `downloaded; "→ Drive" uploads it to Google Drive.`,
    );
  } catch (e) {
    setStatus(`Export failed: ${e}`, true);
  } finally {
    analyzing = false;
    releaseWakeLock();
    analyzeBtn.disabled = false;
    playBtn.disabled = false;
    exportBtn.disabled = false;
    progressBar.hidden = true;
  }
});

driveBtn.addEventListener('click', async () => {
  if (!lastExport) {
    setStatus('No export yet this session — hit "Export MP4" first, then → Drive uploads it.', true);
    return;
  }
  if (!getClientId()) {
    const id = prompt(
      'One-time setup: paste your Google OAuth Client ID.\n\n' +
        'Get one (free) at console.cloud.google.com → APIs & Services →\n' +
        'Credentials → Create OAuth client ID (Web application) with\n' +
        `authorized JavaScript origin ${location.origin}, and enable the\n` +
        'Google Drive API. Details in the README.',
    );
    if (!id?.trim()) return;
    setClientId(id);
  }
  driveBtn.disabled = true;
  progressBar.hidden = false;
  progressBar.value = 0;
  setStatus(`Uploading ${lastExport.name} to Google Drive…`);
  try {
    const file = await uploadToDrive(lastExport.blob, lastExport.name, (frac) => {
      progressBar.value = frac;
      setStatus(`Uploading to Drive… ${(frac * 100).toFixed(0)}%`);
    });
    statusEl.classList.remove('error');
    statusEl.innerHTML = '';
    statusEl.append(`Uploaded ${lastExport.name} to Google Drive — `);
    const link = document.createElement('a');
    link.href = file.link;
    link.target = '_blank';
    link.rel = 'noopener';
    link.textContent = 'open it';
    link.style.color = 'var(--accent)';
    statusEl.append(link);
  } catch (e) {
    setStatus(`Drive upload failed: ${e}`, true);
  } finally {
    driveBtn.disabled = false;
    progressBar.hidden = true;
  }
});

// stride selector is no longer meaningful (analysis is per-frame now)
strideSel.disabled = true;
strideSel.title = 'v2 analyzes every frame';

// restore the last export across reloads so → Drive keeps working
void restoreExport().then((restored) => {
  if (restored && !lastExport) {
    lastExport = restored;
    driveBtn.disabled = false;
  }
});

initDetector()
  .then((ep) => setStatus(`Detector ready (${ep === 'webgpu' ? 'WebGPU 🚀' : 'WASM/CPU — slower'}). Pick a clip.`))
  .catch((e) => setStatus(`${e}`, true));

renderLoop();
window.addEventListener('beforeunload', () => cancelAnimationFrame(rafHandle));
