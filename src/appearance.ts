import type { Detection } from './detect';

// Coarse RGB histogram of the rider's center region. Coarse bins plus a
// center crop keep it robust to small box misalignment while staying
// discriminative enough to tell riders apart by kit/bike color.
const BINS = 4;
const SAMPLE = 24;

export type Appearance = Float32Array;

const canvas = document.createElement('canvas');
canvas.width = SAMPLE;
canvas.height = SAMPLE;
const ctx = canvas.getContext('2d', { willReadFrequently: true })!;

export function captureAppearance(frame: CanvasImageSource, box: Detection): Appearance {
  const w = Math.max(2, box.w * 0.6);
  const h = Math.max(2, box.h * 0.6);
  ctx.drawImage(frame, box.cx - w / 2, box.cy - h / 2, w, h, 0, 0, SAMPLE, SAMPLE);
  const { data } = ctx.getImageData(0, 0, SAMPLE, SAMPLE);
  const hist = new Float32Array(BINS * BINS * BINS);
  const pixels = SAMPLE * SAMPLE;
  for (let i = 0; i < pixels; i++) {
    const r = (data[i * 4] * BINS) >> 8;
    const g = (data[i * 4 + 1] * BINS) >> 8;
    const b = (data[i * 4 + 2] * BINS) >> 8;
    hist[(r * BINS + g) * BINS + b] += 1 / pixels;
  }
  return hist;
}

/** Histogram intersection: 1 = identical color distribution, 0 = disjoint. */
export function appearanceSimilarity(a: Appearance, b: Appearance): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.min(a[i], b[i]);
  return s;
}

/** Slow template adaptation so gradual lighting change doesn't stale it. */
export function blendAppearance(into: Appearance, from: Appearance, alpha: number): void {
  for (let i = 0; i < into.length; i++) into[i] = (1 - alpha) * into[i] + alpha * from[i];
}
