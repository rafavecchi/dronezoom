import type { Detection, Region } from './detect';

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
}

export interface Point {
  x: number;
  y: number;
}

/**
 * Single-target tracker seeded by a user click. Predicts the rider's
 * position with constant velocity, only accepts detections inside a
 * speed-limited gate (so a hiker across the frame can never steal the
 * camera — a far-away detection is a miss, not a switch), and tells the
 * caller where to re-look with a zoomed-in detection pass after misses.
 */
export class Tracker {
  private last: Detection | null = null;
  private lastT = 0;
  private vx = 0;
  private vy = 0;
  private readonly diag: number;

  constructor(
    private seed: Point,
    private srcW: number,
    private srcH: number,
  ) {
    this.diag = Math.hypot(srcW, srcH);
  }

  get acquired(): boolean {
    return this.last !== null;
  }

  predict(t: number): Point {
    if (!this.last) return this.seed;
    // Cap extrapolation so a long miss streak doesn't run the prediction
    // off into the weeds.
    const dt = Math.min(t - this.lastT, 1.5);
    return {
      x: clamp(this.last.cx + this.vx * dt, 0, this.srcW),
      y: clamp(this.last.cy + this.vy * dt, 0, this.srcH),
    };
  }

  /** Zoomed re-detection window around the predicted position. */
  searchRegion(t: number): Region {
    const p = this.predict(t);
    const size = this.last
      ? clamp(this.last.h * 8, 480, this.srcH)
      : Math.max(480, this.srcH / 2);
    return {
      x: clamp(p.x - size / 2, 0, this.srcW - size),
      y: clamp(p.y - size / 2, 0, Math.max(0, this.srcH - size)),
      w: Math.min(size, this.srcW),
      h: Math.min(size, this.srcH),
    };
  }

  match(dets: Detection[], t: number): Detection | null {
    const p = this.predict(t);
    const gate = this.gateRadius(t);
    let best: Detection | null = null;
    let bestCost = Infinity;
    for (const d of dets) {
      const dist = Math.hypot(d.cx - p.x, d.cy - p.y);
      if (dist > gate) continue;
      // Reject size jumps a real rider can't make between samples.
      if (this.last) {
        const ratio = d.h / this.last.h;
        if (ratio < 0.4 || ratio > 2.5) continue;
      }
      const cost = dist / this.diag - d.score * 0.2;
      if (cost < bestCost) {
        bestCost = cost;
        best = d;
      }
    }
    if (best) {
      const dt = t - this.lastT;
      if (this.last && dt > 0 && dt < 1.5) {
        // EMA so one noisy box doesn't slingshot the velocity estimate.
        this.vx = 0.5 * this.vx + (0.5 * (best.cx - this.last.cx)) / dt;
        this.vy = 0.5 * this.vy + (0.5 * (best.cy - this.last.cy)) / dt;
      }
      this.last = best;
      this.lastT = t;
    }
    return best;
  }

  private gateRadius(t: number): number {
    if (!this.last) return 0.15 * this.diag; // first acquisition, around the seed click
    const dt = t - this.lastT;
    const maxSpeed = 0.4 * this.diag; // px/s — generous for a fast descent
    return Math.min(0.3 * this.diag, 0.04 * this.diag + maxSpeed * dt);
  }
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
  const sCx = gaussianSmooth(cx, sigma);
  const sCy = gaussianSmooth(cy, sigma);
  const sPh = gaussianSmooth(ph, sigma * 2);

  const minCropH = srcH / opts.maxZoom;
  const aspect = srcW / srcH;

  return samples.map((s, i) => {
    const cropH = clamp(sPh[i] * opts.padFactor, minCropH, srcH);
    const cropW = cropH * aspect;
    return {
      t: s.t,
      cx: clamp(sCx[i], cropW / 2, srcW - cropW / 2),
      cy: clamp(sCy[i], cropH / 2, srcH - cropH / 2),
      cropW,
      cropH,
    };
  });
}

/** Crop window at an arbitrary time, lerped between path keys. */
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
  const a = path[lo];
  const b = path[hi];
  const f = (t - a.t) / (b.t - a.t);
  return {
    t,
    cx: a.cx + (b.cx - a.cx) * f,
    cy: a.cy + (b.cy - a.cy) * f,
    cropW: a.cropW + (b.cropW - a.cropW) * f,
    cropH: a.cropH + (b.cropH - a.cropH) * f,
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

function gaussianSmooth(values: number[], sigma: number): number[] {
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
