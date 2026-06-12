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
}

/**
 * Pick which detection to follow this frame: nearest to the previous
 * tracked position, score-weighted, so a hiker at the trailside doesn't
 * steal the camera from the rider.
 */
export function pickTarget(dets: Detection[], samples: Sample[], srcW: number, srcH: number): Detection | null {
  if (dets.length === 0) return null;
  const prev = [...samples].reverse().find((s) => s.box)?.box;
  if (!prev) return dets[0]; // dets arrive sorted by score
  const diag = Math.hypot(srcW, srcH);
  let best: Detection | null = null;
  let bestCost = Infinity;
  for (const d of dets) {
    const dist = Math.hypot(d.cx - prev.cx, d.cy - prev.cy) / diag;
    const cost = dist - d.score * 0.3;
    if (cost < bestCost) {
      bestCost = cost;
      best = d;
    }
  }
  return best;
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

  const cx = fillGaps(samples.map((s) => s.box?.cx ?? null));
  const cy = fillGaps(samples.map((s) => s.box?.cy ?? null));
  const ph = fillGaps(samples.map((s) => s.box?.h ?? null));

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
