import type { Detection } from './detect';

export interface Sample {
  t: number;
  box: Detection | null;
}

export interface CropKey {
  t: number;
  cx: number;
  cy: number;
  cropW: number;
  cropH: number;
}

export interface PathOptions {
  /** Crop height as a multiple of the detected rider height. */
  padFactor: number;
  /** Gaussian smoothing of the camera path, in seconds. */
  smoothSigmaSec: number;
  /** Never zoom in tighter than srcH / maxZoom. */
  maxZoom: number;
  /** Spacing between samples, in seconds. */
  sampleInterval: number;
  /**
   * Project a world-coordinate crop center back inside the frame at time
   * t (caller transforms world→frame, clamps, transforms back). Lets the
   * smoothed path absorb frame-boundary limits in advance, instead of the
   * renderer hard-clamping there — which clips the per-frame shake
   * correction and lets shake through right at the edges.
   */
  constrain?: (
    t: number,
    cropW: number,
    cropH: number,
    x: number,
    y: number,
  ) => { x: number; y: number };
  /** Crop-size multiplier at time t (zoom-out-on-pan), >= 1. */
  widen?: (t: number) => number;
}

/**
 * Turn raw per-sample detections into a smooth crop path:
 * fill detection gaps by interpolation, low-pass the center and the
 * subject size (size gets extra smoothing to avoid zoom "breathing"),
 * then derive a clamped, aspect-correct crop window per sample.
 */
export function buildCropPath(
  samples: Sample[],
  srcW: number,
  srcH: number,
  opts: PathOptions,
): CropKey[] {
  const n = samples.length;
  if (n === 0 || samples.every((s) => !s.box)) {
    return [{ t: 0, cx: srcW / 2, cy: srcH / 2, cropW: srcW, cropH: srcH }];
  }

  // Median filter before the Gaussian: a single bad box (shadow, bush)
  // becomes a spike the Gaussian would smear into a visible camera lurch;
  // the median removes it outright.
  const cx = medianFilter(fillGaps(samples.map((s) => s.box?.cx ?? null)));
  const cy = medianFilter(fillGaps(samples.map((s) => s.box?.cy ?? null)));
  const ph = medianFilter(fillGaps(samples.map((s) => s.box?.h ?? null)));

  const sigma = opts.smoothSigmaSec / opts.sampleInterval;
  const sPh = gaussianSmooth(ph, sigma * 2);

  const minCropH = srcH / opts.maxZoom;
  const aspect = srcW / srcH;
  const cropHs = sPh.map((h, i) =>
    clamp(h * opts.padFactor * (opts.widen?.(samples[i].t) ?? 1), minCropH, srcH),
  );

  // "As smooth as possible, subject to the rider staying inside the leash
  // zone": pure Gaussian smoothing lags behind abrupt drone swings and lets
  // the subject slide to the crop edge. Alternate smoothing with a clamp
  // back into the leash window (each round smoothing less), then clamp hard
  // — smooth when the camera is steady, responsive exactly when it swings.
  // Vertical leash is tighter: riders move vertically in fast bursts
  // (drops, terrain) and the 16:9 crop has less vertical headroom.
  const LEASH_X = 0.45;
  const LEASH_Y = 0.35;
  const limX = (i: number) => (cropHs[i] * aspect * LEASH_X) / 2;
  const limY = (i: number) => (cropHs[i] * LEASH_Y) / 2;

  let sCx = gaussianSmooth(cx, sigma);
  let sCy = gaussianSmooth(cy, sigma);
  const applyConstrain = () => {
    if (!opts.constrain) return;
    for (let i = 0; i < samples.length; i++) {
      const c = opts.constrain(samples[i].t, cropHs[i] * aspect, cropHs[i], sCx[i], sCy[i]);
      sCx[i] = c.x;
      sCy[i] = c.y;
    }
  };
  for (let iter = 0; iter < 6; iter++) {
    const s = sigma * Math.pow(0.55, iter + 1);
    sCx = leash(sCx, cx, limX);
    sCy = leash(sCy, cy, limY);
    applyConstrain();
    sCx = gaussianSmooth(sCx, s);
    sCy = gaussianSmooth(sCy, s);
  }
  sCx = leash(sCx, cx, limX);
  sCy = leash(sCy, cy, limY);
  applyConstrain();

  // No frame-boundary clamp here: the path may be in stabilized "world"
  // coordinates — the renderer clamps after adding the per-frame camera
  // offset back.
  return samples.map((s, i) => {
    const cropH = cropHs[i];
    return { t: s.t, cx: sCx[i], cy: sCy[i], cropW: cropH * aspect, cropH };
  });
}

function leash(path: number[], subject: number[], halfWindow: (i: number) => number): number[] {
  return path.map((v, i) =>
    clamp(v, subject[i] - halfWindow(i), subject[i] + halfWindow(i)),
  );
}

/**
 * Crop window at an arbitrary time. Catmull-Rom between path keys —
 * linear interpolation has velocity corners at every key (5–10 per
 * second), which read as bumpiness in the output.
 */
export function cropAt(path: CropKey[], t: number): CropKey {
  if (t <= path[0].t) return path[0];
  const last = path[path.length - 1];
  if (t >= last.t) return last;
  let lo = 0;
  let hi = path.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (path[mid].t <= t) lo = mid;
    else hi = mid;
  }
  const p0 = path[Math.max(0, lo - 1)];
  const p1 = path[lo];
  const p2 = path[hi];
  const p3 = path[Math.min(path.length - 1, hi + 1)];
  const f = (t - p1.t) / (p2.t - p1.t);
  const cr = (a: number, b: number, c: number, d: number) =>
    0.5 *
    (2 * b +
      (-a + c) * f +
      (2 * a - 5 * b + 4 * c - d) * f * f +
      (-a + 3 * b - 3 * c + d) * f * f * f);
  return {
    t,
    cx: cr(p0.cx, p1.cx, p2.cx, p3.cx),
    cy: cr(p0.cy, p1.cy, p2.cy, p3.cy),
    cropW: cr(p0.cropW, p1.cropW, p2.cropW, p3.cropW),
    cropH: cr(p0.cropH, p1.cropH, p2.cropH, p3.cropH),
  };
}

function fillGaps(values: (number | null)[]): number[] {
  const out = values.slice();
  const n = out.length;
  let firstKnown = -1;
  let prevKnown = -1;
  for (let i = 0; i < n; i++) {
    if (out[i] === null) continue;
    if (firstKnown === -1) firstKnown = i;
    if (prevKnown !== -1 && i - prevKnown > 1) {
      const a = out[prevKnown]!;
      const b = out[i]!;
      for (let j = prevKnown + 1; j < i; j++) {
        out[j] = a + ((b - a) * (j - prevKnown)) / (i - prevKnown);
      }
    }
    prevKnown = i;
  }
  for (let i = 0; i < firstKnown; i++) out[i] = out[firstKnown];
  for (let i = prevKnown + 1; i < n; i++) out[i] = out[prevKnown];
  return out as number[];
}

function medianFilter(values: number[], radius = 2): number[] {
  const n = values.length;
  if (n <= 2 * radius) return values.slice();
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - radius);
    const hi = Math.min(n - 1, i + radius);
    const window = values.slice(lo, hi + 1).sort((a, b) => a - b);
    out[i] = window[Math.floor(window.length / 2)];
  }
  return out;
}

export function gaussianSmooth(values: number[], sigma: number): number[] {
  if (sigma <= 0) return values.slice();
  const radius = Math.max(1, Math.ceil(sigma * 3));
  const kernel: number[] = [];
  let sum = 0;
  for (let k = -radius; k <= radius; k++) {
    const w = Math.exp(-(k * k) / (2 * sigma * sigma));
    kernel.push(w);
    sum += w;
  }
  const out = new Array<number>(values.length);
  for (let i = 0; i < values.length; i++) {
    let acc = 0;
    for (let k = -radius; k <= radius; k++) {
      const j = clamp(i + k, 0, values.length - 1);
      acc += values[j] * kernel[k + radius];
    }
    out[i] = acc / sum;
  }
  return out;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}
