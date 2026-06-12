// Per-frame global camera-motion estimation via phase correlation on
// downscaled grayscale frames — the same principle vid.stab uses, running
// live in the browser. Detection sampling (5–10Hz) can never see 30–60Hz
// drone shake; this measures it at full frame rate so the renderer can
// cancel it per frame.

const AW = 512;
const AH = 256;
const N = AW * AH;

export interface CameraPath {
  ts: number[];
  xs: number[];
  ys: number[];
}

const canvas = document.createElement('canvas');
canvas.width = AW;
canvas.height = AH;
const ctx = canvas.getContext('2d', { willReadFrequently: true })!;

const hann = (() => {
  const w = new Float32Array(N);
  for (let y = 0; y < AH; y++) {
    const wy = 0.5 - 0.5 * Math.cos((2 * Math.PI * y) / (AH - 1));
    for (let x = 0; x < AW; x++) {
      const wx = 0.5 - 0.5 * Math.cos((2 * Math.PI * x) / (AW - 1));
      w[y * AW + x] = wx * wy;
    }
  }
  return w;
})();

interface Spec {
  re: Float32Array;
  im: Float32Array;
}

function grabGray(video: HTMLVideoElement): Float32Array {
  ctx.drawImage(video, 0, 0, AW, AH);
  const { data } = ctx.getImageData(0, 0, AW, AH);
  const g = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    g[i] = (data[i * 4] * 0.299 + data[i * 4 + 1] * 0.587 + data[i * 4 + 2] * 0.114) / 255;
  }
  return g;
}

function spectrum(gray: Float32Array): Spec {
  let mean = 0;
  for (let i = 0; i < N; i++) mean += gray[i];
  mean /= N;
  const re = new Float32Array(N);
  const im = new Float32Array(N);
  for (let i = 0; i < N; i++) re[i] = (gray[i] - mean) * hann[i];
  fft2d(re, im, false);
  return { re, im };
}

/**
 * Translation of content from frame A to frame B, in analysis pixels.
 * If b(x) = a(x - d), the normalized cross-power spectrum A·conj(B) has
 * its inverse-FFT peak at -d (mod size), so motion = -peak, with
 * parabolic sub-pixel refinement.
 */
function phaseShift(a: Spec, b: Spec): { dx: number; dy: number } {
  const re = new Float32Array(N);
  const im = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const rr = a.re[i] * b.re[i] + a.im[i] * b.im[i];
    const ii = a.im[i] * b.re[i] - a.re[i] * b.im[i];
    const mag = Math.hypot(rr, ii) + 1e-9;
    re[i] = rr / mag;
    im[i] = ii / mag;
  }
  fft2d(re, im, true);

  let peak = 0;
  for (let i = 1; i < N; i++) if (re[i] > re[peak]) peak = i;
  const px = peak % AW;
  const py = (peak / AW) | 0;

  const at = (x: number, y: number) => re[((y + AH) % AH) * AW + ((x + AW) % AW)];
  const subX = parabolic(at(px - 1, py), at(px, py), at(px + 1, py));
  const subY = parabolic(at(px, py - 1), at(px, py), at(px, py + 1));

  let dx = -(px + subX);
  let dy = -(py + subY);
  if (dx < -AW / 2) dx += AW;
  if (dx > AW / 2) dx -= AW;
  if (dy < -AH / 2) dy += AH;
  if (dy > AH / 2) dy -= AH;
  return { dx, dy };
}

function parabolic(l: number, c: number, r: number): number {
  const denom = l - 2 * c + r;
  if (Math.abs(denom) < 1e-12) return 0;
  const off = (0.5 * (l - r)) / denom;
  return Math.abs(off) <= 1 ? off : 0;
}

/** Test hook: content motion from grayscale frame a to b (analysis px). */
export function estimateShiftGray(a: Float32Array, b: Float32Array): { dx: number; dy: number } {
  return phaseShift(spectrum(a), spectrum(b));
}

/**
 * Play the clip through once, accumulating per-frame global motion into
 * a camera trajectory in source pixels. Dropped frames are harmless: the
 * shift is measured prev→current regardless of the gap.
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

  const sx = video.videoWidth / AW;
  const sy = video.videoHeight / AH;

  video.pause();
  await new Promise<void>((resolve) => {
    video.addEventListener('seeked', () => resolve(), { once: true });
    video.currentTime = 0;
  });

  const ts = [0];
  const xs = [0];
  const ys = [0];
  let prev = spectrum(grabGray(video));
  let cx = 0;
  let cy = 0;
  let lastT = 0;

  await new Promise<void>((resolve) => {
    video.addEventListener('ended', () => resolve(), { once: true });
    const cb = (_now: number, meta: { mediaTime: number }) => {
      if (video.ended) return;
      const t = meta.mediaTime;
      if (t > lastT + 1e-4) {
        const cur = spectrum(grabGray(video));
        const { dx, dy } = phaseShift(prev, cur);
        prev = cur;
        // A shift this large is a scene cut or estimation failure, not shake.
        if (Math.abs(dx) < AW * 0.35 && Math.abs(dy) < AH * 0.35) {
          cx += dx * sx;
          cy += dy * sy;
        }
        ts.push(t);
        xs.push(cx);
        ys.push(cy);
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
  return { ts, xs, ys };
}

/** Camera offset at time t, lerped between per-frame samples. */
export function cameraAt(path: CameraPath, t: number): { x: number; y: number } {
  const { ts, xs, ys } = path;
  if (t <= ts[0]) return { x: xs[0], y: ys[0] };
  const n = ts.length;
  if (t >= ts[n - 1]) return { x: xs[n - 1], y: ys[n - 1] };
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (ts[mid] <= t) lo = mid;
    else hi = mid;
  }
  const f = (t - ts[lo]) / (ts[hi] - ts[lo]);
  return {
    x: xs[lo] + (xs[hi] - xs[lo]) * f,
    y: ys[lo] + (ys[hi] - ys[lo]) * f,
  };
}

// ---- FFT ----

function fft2d(re: Float32Array, im: Float32Array, inverse: boolean): void {
  for (let y = 0; y < AH; y++) {
    fft1d(re.subarray(y * AW, (y + 1) * AW), im.subarray(y * AW, (y + 1) * AW), inverse);
  }
  const tr = new Float32Array(AH);
  const ti = new Float32Array(AH);
  for (let x = 0; x < AW; x++) {
    for (let y = 0; y < AH; y++) {
      tr[y] = re[y * AW + x];
      ti[y] = im[y * AW + x];
    }
    fft1d(tr, ti, inverse);
    for (let y = 0; y < AH; y++) {
      re[y * AW + x] = tr[y];
      im[y * AW + x] = ti[y];
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
