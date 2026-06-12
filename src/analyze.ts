// Single analysis pass: seek-step every frame, estimate the camera
// similarity transform, track the rider as a motion blob in the
// shake-aligned frame difference, and re-anchor identity with YOLO
// every ~0.2s. Runs forward then backward from the seeded frame.

import { detectPersons } from './detect';
import {
  estimatePair,
  frameSpectra,
  invert,
  GW,
  GH,
  type Affine,
  type FrameIncrement,
} from './motion';
import type { Sample } from './path';
import { Tracker, type Point } from './tracker';

const DS = 2; // blob work at half analysis res
const BW = GW / DS;
const BH = GH / DS;
const SEARCH = 40; // blob search radius, ds px (≈80 analysis px)

export interface AnalysisResult {
  fps: number;
  /** frame i -> i+1 content motion, source px; length = frames-1 */
  incs: FrameIncrement[];
  /** per-frame rider position/size in source px */
  blobX: Float64Array;
  blobY: Float64Array;
  blobH: Float64Array;
  blobFound: Uint8Array;
  /** YOLO samples kept for the overlay */
  samples: Sample[];
  frames: number;
}

const grab = document.createElement('canvas');
grab.width = GW;
grab.height = GH;
const grabCtx = grab.getContext('2d', { willReadFrequently: true })!;

function grabGray(video: HTMLVideoElement): Float32Array {
  grabCtx.drawImage(video, 0, 0, GW, GH);
  const { data } = grabCtx.getImageData(0, 0, GW, GH);
  const g = new Float32Array(GW * GH);
  for (let i = 0; i < g.length; i++) {
    g[i] = data[i * 4] * 0.299 + data[i * 4 + 1] * 0.587 + data[i * 4 + 2] * 0.114;
  }
  return g;
}

/** out(p) = src(M⁻¹p), bilinear — same content motion as the camera. */
function alignedDiffDs(prev: Float32Array, cur: Float32Array, M: Affine): Float32Array {
  const Mi = invert(M);
  const out = new Float32Array(BW * BH);
  for (let y = 0; y < BH; y++) {
    for (let x = 0; x < BW; x++) {
      // average the 2x2 cur block, sample prev at its warped center
      const gx = x * DS + 0.5;
      const gy = y * DS + 0.5;
      const sx = Mi[0] * gx + Mi[1] * gy + Mi[2];
      const sy = Mi[3] * gx + Mi[4] * gy + Mi[5];
      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      if (x0 < 0 || y0 < 0 || x0 >= GW - 1 || y0 >= GH - 1) continue;
      const fx = sx - x0;
      const fy = sy - y0;
      const p =
        prev[y0 * GW + x0] * (1 - fx) * (1 - fy) +
        prev[y0 * GW + x0 + 1] * fx * (1 - fy) +
        prev[(y0 + 1) * GW + x0] * (1 - fx) * fy +
        prev[(y0 + 1) * GW + x0 + 1] * fx * fy;
      const c =
        0.25 *
        (cur[gy0(y) * GW + gx0(x)] +
          cur[gy0(y) * GW + gx0(x) + 1] +
          cur[(gy0(y) + 1) * GW + gx0(x)] +
          cur[(gy0(y) + 1) * GW + gx0(x) + 1]);
      out[y * BW + x] = Math.abs(c - p);
    }
  }
  return out;
}

function gx0(x: number): number {
  return x * DS;
}
function gy0(y: number): number {
  return y * DS;
}

function boxBlur(src: Float32Array, w: number, h: number, r: number): Float32Array {
  // two-pass separable box blur (approx gaussian when applied twice)
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  const norm = 1 / (2 * r + 1);
  for (let y = 0; y < h; y++) {
    let acc = 0;
    for (let x = -r; x <= r; x++) acc += src[y * w + Math.min(w - 1, Math.max(0, x))];
    for (let x = 0; x < w; x++) {
      tmp[y * w + x] = acc * norm;
      const add = Math.min(w - 1, x + r + 1);
      const sub = Math.max(0, x - r);
      acc += src[y * w + add] - src[y * w + sub];
    }
  }
  for (let x = 0; x < w; x++) {
    let acc = 0;
    for (let y = -r; y <= r; y++) acc += tmp[Math.min(h - 1, Math.max(0, y)) * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = acc * norm;
      const add = Math.min(h - 1, y + r + 1);
      const sub = Math.max(0, y - r);
      acc += tmp[add * w + x] - tmp[sub * w + x];
    }
  }
  return out;
}

interface BlobState {
  x: number; // ds px
  y: number;
  vx: number;
  vy: number;
  h: number; // ds px
  resHistory: number[];
  miss: number;
}

/**
 * The camera moved: carry the prediction (and velocity) with it. During
 * a whip pan the rider's frame position moves with the camera, not with
 * his own ground motion — without this, every fast pan loses the track.
 */
function carryByCamera(state: BlobState, M: Affine) {
  const ax = state.x * DS;
  const ay = state.y * DS;
  state.x = (M[0] * ax + M[1] * ay + M[2]) / DS;
  state.y = (M[3] * ax + M[4] * ay + M[5]) / DS;
  const vx = M[0] * state.vx + M[1] * state.vy;
  const vy = M[3] * state.vx + M[4] * state.vy;
  state.vx = vx;
  state.vy = vy;
}

function blobStep(
  state: BlobState,
  diff: Float32Array,
): { found: boolean } {
  const m = 4;
  for (let y = 0; y < BH; y++) {
    for (let x = 0; x < BW; x++) {
      if (x < m || y < m || x >= BW - m || y >= BH - m) diff[y * BW + x] = 0;
    }
  }
  let mean = 0;
  for (let i = 0; i < diff.length; i++) mean += diff[i];
  mean /= diff.length;
  state.resHistory.push(mean);
  if (state.resHistory.length > 90) state.resHistory.shift();
  const sorted = [...state.resHistory].sort((a, b) => a - b);
  const baseRes = sorted[sorted.length >> 1];
  const alignBad = mean > Math.max(2.2 * baseRes, 25);

  // center-surround: compact blobs pop, parallax bands cancel
  const b1a = boxBlur(diff, BW, BH, 1);
  const small = boxBlur(b1a, BW, BH, 1);
  const b2a = boxBlur(diff, BW, BH, 4);
  const large = boxBlur(b2a, BW, BH, 4);

  const px = Math.min(Math.max(state.x + state.vx, 0), BW - 1);
  const py = Math.min(Math.max(state.y + state.vy, 0), BH - 1);
  let best = -Infinity;
  let bx = 0;
  let by = 0;
  // search widens with the miss streak so a lost track can re-acquire
  const sEff = Math.min(SEARCH * (1 + state.miss / 15), 150);
  const inv2s2 = 1 / (2 * sEff * sEff);
  for (let y = m; y < BH - m; y++) {
    const dy2 = (y - py) * (y - py);
    if (dy2 > 9 * sEff * sEff) continue;
    for (let x = m; x < BW - m; x++) {
      const resp = small[y * BW + x] - 0.8 * large[y * BW + x];
      const d2 = (x - px) * (x - px) + dy2;
      const s = resp * Math.exp(-d2 * inv2s2);
      if (s > best) {
        best = s;
        bx = x;
        by = y;
      }
    }
  }
  const quality = small[by * BW + bx] - 0.8 * large[by * BW + bx];
  const found = quality > 4 && !alignBad;
  if (found) {
    // centroid refine
    const r = 6;
    let sw = 0;
    let sx = 0;
    let sy = 0;
    for (let y = Math.max(by - r, 0); y < Math.min(by + r, BH); y++) {
      for (let x = Math.max(bx - r, 0); x < Math.min(bx + r, BW); x++) {
        const v = Math.max(0, small[y * BW + x] - 0.8 * large[y * BW + x]);
        sw += v;
        sx += v * x;
        sy += v * y;
      }
    }
    const nx = sw > 0 ? sx / sw : bx;
    const ny = sw > 0 ? sy / sw : by;
    state.vx = 0.6 * state.vx + 0.4 * (nx - state.x);
    state.vy = 0.6 * state.vy + 0.4 * (ny - state.y);
    state.x = nx;
    state.y = ny;
    state.miss = 0;
    // height: vertical extent of diff > 0.4*peak in a column band
    const peakV = diff[by * BW + bx];
    let hh = 0;
    for (let y = Math.max(by - 12, 0); y < Math.min(by + 12, BH); y++) {
      let hit = false;
      for (let x = Math.max(bx - 2, 0); x < Math.min(bx + 3, BW); x++) {
        if (diff[y * BW + x] > peakV * 0.4) hit = true;
      }
      if (hit) hh++;
    }
    state.h = Math.max(3, hh);
  } else {
    state.miss++;
    state.vx *= 0.9;
    state.vy *= 0.9;
    state.x = Math.min(Math.max(state.x + state.vx, 0), BW - 1);
    state.y = Math.min(Math.max(state.y + state.vy, 0), BH - 1);
  }
  return { found };
}

function snapFps(raw: number): number {
  const common = [23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60];
  let best = 30;
  for (const f of common) if (Math.abs(f - raw) < Math.abs(best - raw)) best = f;
  return best;
}

export async function probeFps(video: HTMLVideoElement): Promise<number> {
  const rvfc = (
    video as unknown as {
      requestVideoFrameCallback?: (cb: (now: number, meta: { mediaTime: number }) => void) => number;
    }
  ).requestVideoFrameCallback?.bind(video);
  if (!rvfc) return 30;
  video.pause();
  video.currentTime = 0;
  const gaps: number[] = [];
  let last = -1;
  const result = await new Promise<number>((resolve) => {
    const timeout = setTimeout(() => resolve(30), 3000);
    const cb = (_n: number, meta: { mediaTime: number }) => {
      if (last >= 0 && meta.mediaTime > last) gaps.push(meta.mediaTime - last);
      last = meta.mediaTime;
      if (gaps.length >= 10) {
        clearTimeout(timeout);
        gaps.sort((a, b) => a - b);
        resolve(snapFps(1 / gaps[gaps.length >> 1]));
        return;
      }
      rvfc(cb);
    };
    rvfc(cb);
    void video.play();
  });
  video.pause();
  return result;
}

export async function runAnalysis(
  video: HTMLVideoElement,
  seed: Point,
  seedT: number,
  fps: number,
  onProgress: (done: number, total: number, roiHits: number, blobHits: number) => void,
): Promise<AnalysisResult> {
  const srcW = video.videoWidth;
  const srcH = video.videoHeight;
  const toSrc = srcW / GW; // uniform for 16:9
  const frames = Math.max(2, Math.floor(video.duration * fps));
  const seedIdx = Math.min(frames - 1, Math.max(0, Math.round(seedT * fps)));
  const detEvery = Math.max(1, Math.round(0.2 * fps));

  const incs: FrameIncrement[] = new Array(frames - 1)
    .fill(null)
    .map(() => ({ dtheta: 0, dlogs: 0, dtx: 0, dty: 0, quality: 0 }));
  const blobX = new Float64Array(frames);
  const blobY = new Float64Array(frames);
  const blobH = new Float64Array(frames);
  const blobFound = new Uint8Array(frames);
  const samples: Sample[] = [];

  const seekTo = (t: number) =>
    new Promise<void>((resolve) => {
      video.addEventListener('seeked', () => resolve(), { once: true });
      video.currentTime = Math.min(t, Math.max(0, video.duration - 0.001));
    });

  let done = 0;
  let roiHits = 0;
  let blobHits = 0;

  const runSegment = async (from: number, to: number, step: 1 | -1) => {
    const tracker = new Tracker({ ...seed }, srcW, srcH);
    const blob: BlobState = {
      x: seed.x / toSrc / DS,
      y: seed.y / toSrc / DS,
      vx: 0,
      vy: 0,
      h: 12,
      resHistory: [],
      miss: 0,
    };
    let prevGray: Float32Array | null = null;
    let prevSpec: ReturnType<typeof frameSpectra> | null = null;
    for (let k = from; step > 0 ? k <= to : k >= to; k += step) {
      await seekTo((k + 0.5) / fps);
      const gray = grabGray(video);
      const spec = frameSpectra(gray);
      if (prevGray && prevSpec) {
        const { M, quality } = estimatePair(prevSpec, spec);
        // store increment oriented forward (frame i -> i+1)
        const Mf = step > 0 ? M : invert(M);
        const idx = step > 0 ? k - 1 : k;
        const th = Math.atan2(Mf[3], Mf[0]);
        const sc = Math.hypot(Mf[0], Mf[3]);
        incs[idx] = {
          dtheta: th,
          dlogs: Math.log(Math.max(sc, 1e-6)),
          dtx: Mf[2] * toSrc,
          dty: Mf[5] * toSrc,
          quality,
        };
        carryByCamera(blob, M);
        const diff = alignedDiffDs(prevGray, gray, M);
        const { found } = blobStep(blob, diff);
        if (found) blobHits++;
        blobFound[k] = found ? 1 : 0;
      }
      // YOLO re-anchor on the sample grid
      if (k % detEvery === 0) {
        const t = k / fps;
        const sizePx = Math.min(Math.max(blob.h * DS * toSrc * 8, 480), srcH);
        const region = {
          x: Math.min(Math.max(blob.x * DS * toSrc - sizePx / 2, 0), Math.max(0, srcW - sizePx)),
          y: Math.min(Math.max(blob.y * DS * toSrc - sizePx / 2, 0), Math.max(0, srcH - sizePx)),
          w: Math.min(sizePx, srcW),
          h: Math.min(sizePx, srcH),
        };
        let box = tracker.match(video, await detectPersons(video, srcW, srcH, region), t);
        if (!box) box = tracker.match(video, await detectPersons(video, srcW, srcH), t);
        if (box) {
          roiHits++;
          const bx = box.cx / toSrc / DS;
          const by = box.cy / toSrc / DS;
          if (Math.hypot(bx - blob.x, by - blob.y) < 100 / DS) {
            blob.x = 0.7 * blob.x + 0.3 * bx;
            blob.y = 0.7 * blob.y + 0.3 * by;
            blob.h = Math.max(blob.h, box.h / toSrc / DS);
          }
        }
        samples.push({ t, box });
      }
      blobX[k] = blob.x * DS * toSrc;
      blobY[k] = blob.y * DS * toSrc;
      blobH[k] = blob.h * DS * toSrc;
      prevGray = gray;
      prevSpec = spec;
      done++;
      onProgress(done, frames, roiHits, blobHits);
    }
  };

  await runSegment(seedIdx, frames - 1, 1);
  if (seedIdx > 0) await runSegment(seedIdx, 0, -1);

  samples.sort((a, b) => a.t - b.t);
  return { fps, incs, blobX, blobY, blobH, blobFound, samples, frames };
}
