// Per-frame global camera-motion estimation via phase correlation on
// downscaled grayscale frames — the same principle vid.stab uses, running
// live in the browser. Detection sampling (5–10Hz) can never see 30–60Hz
// drone shake; this measures it at full frame rate so the renderer can
// cancel it per frame.
//
// Translation: phase correlation of the full downscaled frame.
// Rotation (roll): the frame's left and right halves are correlated
// separately — roll shows up as opposite vertical motion in the two
// halves, and the differential gives the angle.

const AW = 512;
const AH = 256;
const HW = 256; // half width

export interface CameraPath {
  ts: number[];
  xs: number[];
  ys: number[];
  /** Cumulative roll, radians, about the frame center. */
  rs: number[];
}

const canvas = document.createElement('canvas');
canvas.width = AW;
canvas.height = AH;
const ctx = canvas.getContext('2d', { willReadFrequently: true })!;

const hannCache = new Map<string, Float32Array>();

function getHann(w: number, h: number): Float32Array {
  const key = `${w}x${h}`;
  let win = hannCache.get(key);
  if (!win) {
    win = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      const wy = 0.5 - 0.5 * Math.cos((2 * Math.PI * y) / (h - 1));
      for (let x = 0; x < w; x++) {
        const wx = 0.5 - 0.5 * Math.cos((2 * Math.PI * x) / (w - 1));
        win[y * w + x] = wx * wy;
      }
    }
    hannCache.set(key, win);
  }
  return win;
}

interface Spec {
  re: Float32Array;
  im: Float32Array;
  w: number;
  h: number;
}

function grabGray(video: HTMLVideoElement): Float32Array {
  ctx.drawImage(video, 0, 0, AW, AH);
  const { data } = ctx.getImageData(0, 0, AW, AH);
  const g = new Float32Array(AW * AH);
  for (let i = 0; i < g.length; i++) {
    g[i] = (data[i * 4] * 0.299 + data[i * 4 + 1] * 0.587 + data[i * 4 + 2] * 0.114) / 255;
  }
  return g;
}

function extractHalf(gray: Float32Array, right: boolean): Float32Array {
  const out = new Float32Array(HW * AH);
  const xOff = right ? HW : 0;
  for (let y = 0; y < AH; y++) {
    for (let x = 0; x < HW; x++) {
      out[y * HW + x] = gray[y * AW + x + xOff];
    }
  }
  return out;
}

function spectrum(gray: Float32Array, w: number, h: number): Spec {
  const n = w * h;
  const hann = getHann(w, h);
  let mean = 0;
  for (let i = 0; i < n; i++) mean += gray[i];
  mean /= n;
  const re = new Float32Array(n);
  const im = new Float32Array(n);
  for (let i = 0; i < n; i++) re[i] = (gray[i] - mean) * hann[i];
  fft2d(re, im, w, h, false);
  return { re, im, w, h };
}

/**
 * Translation of content from frame A to frame B, in analysis pixels.
 * If b(x) = a(x - d), the normalized cross-power spectrum A·conj(B) has
 * its inverse-FFT peak at -d (mod size), so motion = -peak, with
 * parabolic sub-pixel refinement.
 */
function phaseShift(a: Spec, b: Spec): { dx: number; dy: number } {
  const { w, h } = a;
  const n = w * h;
  const re = new Float32Array(n);
  const im = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const rr = a.re[i] * b.re[i] + a.im[i] * b.im[i];
    const ii = a.im[i] * b.re[i] - a.re[i] * b.im[i];
    const mag = Math.hypot(rr, ii) + 1e-9;
    re[i] = rr / mag;
    im[i] = ii / mag;
  }
  fft2d(re, im, w, h, true);

  let peak = 0;
  for (let i = 1; i < n; i++) if (re[i] > re[peak]) peak = i;
  const px = peak % w;
  const py = (peak / w) | 0;

  const at = (x: number, y: number) => re[((y + h) % h) * w + ((x + w) % w)];
  const subX = parabolic(at(px - 1, py), at(px, py), at(px + 1, py));
  const subY = parabolic(at(px, py - 1), at(px, py), at(px, py + 1));

  let dx = -(px + subX);
  let dy = -(py + subY);
  if (dx < -w / 2) dx += w;
  if (dx > w / 2) dx -= w;
  if (dy < -h / 2) dy += h;
  if (dy > h / 2) dy -= h;
  return { dx, dy };
}

function parabolic(l: number, c: number, r: number): number {
  const denom = l - 2 * c + r;
  if (Math.abs(denom) < 1e-12) return 0;
  const off = (0.5 * (l - r)) / denom;
  return Math.abs(off) <= 1 ? off : 0;
}

/** Test hook: content motion from grayscale frame a to b (analysis px). */
export function estimateShiftGray(
  a: Float32Array,
  b: Float32Array,
  w = AW,
  h = AH,
): { dx: number; dy: number } {
  return phaseShift(spectrum(a, w, h), spectrum(b, w, h));
}

/**
 * Play the clip through once, accumulating per-frame global motion into
 * a camera trajectory (translation in source pixels, roll in radians).
 * Dropped frames are harmless: shifts are measured prev→current
 * regardless of the gap.
 */
export async function estimateCameraPath(
  video: HTMLVideoElement,
  onProgress: (t: number) => void,
): Promise<CameraPath> {
  const rvfc = (
    video as unknown as {
      requestVideoFrameCallback?: (cb: (now: number, meta: { mediaTime: number }) => void) => number;
    }
  ).requestVideoFrameCallback?.bind(video);
  if (!rvfc) throw new Error('requestVideoFrameCallback not supported in this browser');

  const srcW = video.videoWidth;
  const srcH = video.videoHeight;
  const sx = srcW / AW;
  const sy = srcH / AH;

  video.pause();
  await new Promise<void>((resolve) => {
    video.addEventListener('seeked', () => resolve(), { once: true });
    video.currentTime = 0;
  });

  const ts = [0];
  const xs = [0];
  const ys = [0];
  const rs = [0];
  let gray = grabGray(video);
  let prev = spectrum(gray, AW, AH);
  let prevL = spectrum(extractHalf(gray, false), HW, AH);
  let prevR = spectrum(extractHalf(gray, true), HW, AH);
  let cx = 0;
  let cy = 0;
  let cr = 0;
  let lastT = 0;

  await new Promise<void>((resolve) => {
    video.addEventListener('ended', () => resolve(), { once: true });
    const cb = (_now: number, meta: { mediaTime: number }) => {
      if (video.ended) return;
      const t = meta.mediaTime;
      if (t > lastT + 1e-4) {
        gray = grabGray(video);
        const cur = spectrum(gray, AW, AH);
        const curL = spectrum(extractHalf(gray, false), HW, AH);
        const curR = spectrum(extractHalf(gray, true), HW, AH);
        const { dx, dy } = phaseShift(prev, cur);
        const shL = phaseShift(prevL, curL);
        const shR = phaseShift(prevR, curR);
        prev = cur;
        prevL = curL;
        prevR = curR;

        // A shift this large is a scene cut or estimation failure, not shake.
        if (Math.abs(dx) < AW * 0.35 && Math.abs(dy) < AH * 0.35) {
          cx += dx * sx;
          cy += dy * sy;
        }
        // Roll: differential vertical motion of the half frames, whose
        // centers sit srcW/2 apart in source pixels.
        const dTheta = ((shR.dy - shL.dy) * sy) / (srcW / 2);
        if (Math.abs(dTheta) < 0.05 && Math.abs(shL.dy) < AH * 0.3 && Math.abs(shR.dy) < AH * 0.3) {
          cr += dTheta;
        }
        ts.push(t);
        xs.push(cx);
        ys.push(cy);
        rs.push(cr);
        lastT = t;
        onProgress(t);
      }
      rvfc(cb);
    };
    rvfc(cb);
    video.playbackRate = 1;
    void video.play();
  });
  video.pause();
  return { ts, xs, ys, rs };
}

/** Linear interpolation over a (sorted ts, values) series. */
export function lerpSeries(ts: number[], vals: number[], t: number): number {
  if (t <= ts[0]) return vals[0];
  const n = ts.length;
  if (t >= ts[n - 1]) return vals[n - 1];
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (ts[mid] <= t) lo = mid;
    else hi = mid;
  }
  const f = (t - ts[lo]) / (ts[hi] - ts[lo]);
  return vals[lo] + (vals[hi] - vals[lo]) * f;
}

/** Camera offset at time t, lerped between per-frame samples. */
export function cameraAt(path: CameraPath, t: number): { x: number; y: number; r: number } {
  return {
    x: lerpSeries(path.ts, path.xs, t),
    y: lerpSeries(path.ts, path.ys, t),
    r: lerpSeries(path.ts, path.rs, t),
  };
}

// ---- FFT ----

function fft2d(re: Float32Array, im: Float32Array, w: number, h: number, inverse: boolean): void {
  for (let y = 0; y < h; y++) {
    fft1d(re.subarray(y * w, (y + 1) * w), im.subarray(y * w, (y + 1) * w), inverse);
  }
  const tr = new Float32Array(h);
  const ti = new Float32Array(h);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      tr[y] = re[y * w + x];
      ti[y] = im[y * w + x];
    }
    fft1d(tr, ti, inverse);
    for (let y = 0; y < h; y++) {
      re[y * w + x] = tr[y];
      im[y * w + x] = ti[y];
    }
  }
}

function fft1d(re: Float32Array, im: Float32Array, inverse: boolean): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((inverse ? 2 : -2) * Math.PI) / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    const half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cRe = 1;
      let cIm = 0;
      for (let j = 0; j < half; j++) {
        const uRe = re[i + j];
        const uIm = im[i + j];
        const vRe = re[i + j + half] * cRe - im[i + j + half] * cIm;
        const vIm = re[i + j + half] * cIm + im[i + j + half] * cRe;
        re[i + j] = uRe + vRe;
        im[i + j] = uIm + vIm;
        re[i + j + half] = uRe - vRe;
        im[i + j + half] = uIm - vIm;
        const nRe = cRe * wRe - cIm * wIm;
        cIm = cRe * wIm + cIm * wRe;
        cRe = nRe;
      }
    }
  }
  if (inverse) {
    for (let i = 0; i < n; i++) {
      re[i] /= n;
      im[i] /= n;
    }
  }
}
