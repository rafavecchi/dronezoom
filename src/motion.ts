// Camera-motion engine: grid phase correlation -> per-frame similarity
// transform -> incremental shake residuals. Validated offline in
// analysis/v4.py against real footage (see ARCHITECTURE.md).

// ---- affine 2x3: [m00, m01, m02, m10, m11, m12], row-major ----

export type Affine = [number, number, number, number, number, number];

export const IDENTITY: Affine = [1, 0, 0, 0, 1, 0];

export function compose(M: Affine, C: Affine): Affine {
  // apply C then M
  return [
    M[0] * C[0] + M[1] * C[3],
    M[0] * C[1] + M[1] * C[4],
    M[0] * C[2] + M[1] * C[5] + M[2],
    M[3] * C[0] + M[4] * C[3],
    M[3] * C[1] + M[4] * C[4],
    M[3] * C[2] + M[4] * C[5] + M[5],
  ];
}

export function invert(M: Affine): Affine {
  const det = M[0] * M[4] - M[1] * M[3];
  const ia = M[4] / det;
  const ib = -M[1] / det;
  const ic = -M[3] / det;
  const id = M[0] / det;
  return [ia, ib, -(ia * M[2] + ib * M[5]), ic, id, -(ic * M[2] + id * M[5])];
}

export function apply(M: Affine, x: number, y: number): { x: number; y: number } {
  return { x: M[0] * x + M[1] * y + M[2], y: M[3] * x + M[4] * y + M[5] };
}

export function decompose(M: Affine): { theta: number; scale: number; tx: number; ty: number } {
  return { theta: Math.atan2(M[3], M[0]), scale: Math.hypot(M[0], M[3]), tx: M[2], ty: M[5] };
}

export function fromParams(theta: number, scale: number, tx: number, ty: number): Affine {
  const a = scale * Math.cos(theta);
  const b = scale * Math.sin(theta);
  return [a, -b, tx, b, a, ty];
}

// ---- analysis dimensions ----

// 960x540 grab = uniform scale from 16:9 sources, so a similarity in
// analysis px stays a similarity in source px.
export const GW = 960;
export const GH = 540;
const WIN = 256;
const GRID: Array<[number, number]> = [
  [21, 0],
  [352, 0],
  [683, 0],
  [21, 284],
  [352, 284],
  [683, 284],
];

export interface FrameIncrement {
  dtheta: number;
  dlogs: number;
  dtx: number; // source px
  dty: number;
  quality: number; // mean correlation peak of inlier windows
}

interface Spec {
  re: Float32Array;
  im: Float32Array;
}

const hannCache = new Map<number, Float32Array>();

function getHann(n: number): Float32Array {
  let w = hannCache.get(n);
  if (!w) {
    w = new Float32Array(n * n);
    for (let y = 0; y < n; y++) {
      const wy = 0.5 - 0.5 * Math.cos((2 * Math.PI * y) / (n - 1));
      for (let x = 0; x < n; x++) {
        w[y * n + x] = wy * (0.5 - 0.5 * Math.cos((2 * Math.PI * x) / (n - 1)));
      }
    }
    hannCache.set(n, w);
  }
  return w;
}

function windowSpectrum(gray: Float32Array, stride: number, x0: number, y0: number): Spec {
  const n = WIN;
  const hann = getHann(n);
  const re = new Float32Array(n * n);
  const im = new Float32Array(n * n);
  let mean = 0;
  for (let y = 0; y < n; y++) {
    const row = (y0 + y) * stride + x0;
    for (let x = 0; x < n; x++) mean += gray[row + x];
  }
  mean /= n * n;
  for (let y = 0; y < n; y++) {
    const row = (y0 + y) * stride + x0;
    for (let x = 0; x < n; x++) re[y * n + x] = (gray[row + x] - mean) * hann[y * n + x];
  }
  fft2d(re, im, n, n, false);
  return { re, im };
}

function phaseShift(a: Spec, b: Spec, n: number): { dx: number; dy: number; peak: number } {
  const N = n * n;
  const re = new Float32Array(N);
  const im = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const rr = a.re[i] * b.re[i] + a.im[i] * b.im[i];
    const ii = a.im[i] * b.re[i] - a.re[i] * b.im[i];
    const mag = Math.hypot(rr, ii) + 1e-9;
    re[i] = rr / mag;
    im[i] = ii / mag;
  }
  fft2d(re, im, n, n, true);
  let peak = 0;
  for (let i = 1; i < N; i++) if (re[i] > re[peak]) peak = i;
  const px = peak % n;
  const py = (peak / n) | 0;
  const at = (x: number, y: number) => re[((y + n) % n) * n + ((x + n) % n)];
  const parab = (l: number, c: number, r: number) => {
    const d = l - 2 * c + r;
    if (Math.abs(d) < 1e-12) return 0;
    const o = (0.5 * (l - r)) / d;
    return Math.abs(o) <= 1 ? o : 0;
  };
  let dx = -(px + parab(at(px - 1, py), at(px, py), at(px + 1, py)));
  let dy = -(py + parab(at(px, py - 1), at(px, py), at(px, py + 1)));
  if (dx < -n / 2) dx += n;
  if (dx > n / 2) dx -= n;
  if (dy < -n / 2) dy += n;
  if (dy > n / 2) dy -= n;
  return { dx, dy, peak: re[peak] };
}

/** Per-window spectra of a grabbed frame, reused for the next pair. */
export function frameSpectra(gray: Float32Array): Spec[] {
  return GRID.map(([x0, y0]) => windowSpectrum(gray, GW, x0, y0));
}

function fitSimilarity(
  centers: Array<[number, number]>,
  shifts: Array<[number, number]>,
  weights: number[],
): Affine {
  // weighted least squares for p+d = A p + t, A=[[a,-b],[b,a]]
  // normal equations over unknowns (a, b, tx, ty)
  let s00 = 0, s01 = 0, s02 = 0, s03 = 0;
  let s11 = 0, s12 = 0, s13 = 0, s22 = 0, s23 = 0, s33 = 0;
  let r0 = 0, r1 = 0, r2 = 0, r3 = 0;
  for (let i = 0; i < centers.length; i++) {
    const [px, py] = centers[i];
    const [dx, dy] = shifts[i];
    const w = weights[i];
    const ux = px + dx;
    const uy = py + dy;
    // row1: a*px - b*py + tx = ux ; row2: a*py + b*px + ty = uy
    s00 += w * (px * px + py * py);
    s01 += 0; // a-b cross terms cancel: px*(-py) + py*(px) = 0
    s02 += w * px;
    s03 += w * py;
    s11 += w * (py * py + px * px);
    s12 += w * -py;
    s13 += w * px;
    s22 += w;
    s23 += 0;
    s33 += w;
    r0 += w * (px * ux + py * uy);
    r1 += w * (-py * ux + px * uy);
    r2 += w * ux;
    r3 += w * uy;
  }
  // solve 4x4 (symmetric, sparse) via Gaussian elimination
  const A = [
    [s00, s01, s02, s03, r0],
    [s01, s11, s12, s13, r1],
    [s02, s12, s22, s23, r2],
    [s03, s13, s23, s33, r3],
  ];
  for (let c = 0; c < 4; c++) {
    let piv = c;
    for (let r = c + 1; r < 4; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
    [A[c], A[piv]] = [A[piv], A[c]];
    if (Math.abs(A[c][c]) < 1e-12) return IDENTITY;
    for (let r = 0; r < 4; r++) {
      if (r === c) continue;
      const f = A[r][c] / A[c][c];
      for (let k = c; k < 5; k++) A[r][k] -= f * A[c][k];
    }
  }
  const a = A[0][4] / A[0][0];
  const b = A[1][4] / A[1][1];
  const tx = A[2][4] / A[2][2];
  const ty = A[3][4] / A[3][3];
  return [a, -b, tx, b, a, ty];
}

/**
 * Similarity transform (analysis px) for a frame pair from per-window
 * phase correlations, with outlier rejection and sanity clamping.
 */
export function estimatePair(prev: Spec[], cur: Spec[]): { M: Affine; quality: number } {
  const centers: Array<[number, number]> = [];
  const shifts: Array<[number, number]> = [];
  const weights: number[] = [];
  for (let i = 0; i < GRID.length; i++) {
    const { dx, dy, peak } = phaseShift(prev[i], cur[i], WIN);
    if (peak > 0.03 && Math.abs(dx) < WIN * 0.3 && Math.abs(dy) < WIN * 0.3) {
      centers.push([GRID[i][0] + WIN / 2, GRID[i][1] + WIN / 2]);
      shifts.push([dx, dy]);
      weights.push(peak);
    }
  }
  let M: Affine = IDENTITY;
  let inliers = 0;
  if (centers.length >= 3) {
    M = fitSimilarity(centers, shifts, weights);
    const kc: Array<[number, number]> = [];
    const ks: Array<[number, number]> = [];
    const kw: number[] = [];
    for (let i = 0; i < centers.length; i++) {
      const p = apply(M, centers[i][0], centers[i][1]);
      const res = Math.hypot(
        p.x - (centers[i][0] + shifts[i][0]),
        p.y - (centers[i][1] + shifts[i][1]),
      );
      if (res < 3) {
        kc.push(centers[i]);
        ks.push(shifts[i]);
        kw.push(weights[i]);
      }
    }
    inliers = kc.length;
    if (kc.length >= 3 && kc.length < centers.length) M = fitSimilarity(kc, ks, kw);
  }
  // Sanity: garbage fits (whip pans, low texture) would poison the
  // chain. High-consensus fits (>=4 inlier windows) are trusted to much
  // larger angles — aggressive drone yaw really does 2-3 deg/frame, and
  // clamping those frames leaves visible rotational jolts uncorrected.
  const d = decompose(M);
  const limTheta = inliers >= 4 ? 0.12 : 0.035;
  const limScale = inliers >= 4 ? 0.08 : 0.03;
  if (
    Math.abs(d.theta) > limTheta ||
    Math.abs(d.scale - 1) > limScale ||
    Math.hypot(d.tx, d.ty) > 200 ||
    centers.length < 3
  ) {
    let mdx = 0;
    let mdy = 0;
    if (shifts.length) {
      const xs = shifts.map((s) => s[0]).sort((p, q) => p - q);
      const ys = shifts.map((s) => s[1]).sort((p, q) => p - q);
      mdx = xs[xs.length >> 1];
      mdy = ys[ys.length >> 1];
    }
    M = [1, 0, mdx, 0, 1, mdy];
  }
  const quality = weights.length ? weights.reduce((s, v) => s + v, 0) / weights.length : 0;
  return { M, quality };
}

/**
 * Shake residual per frame from increments: smooth the small increment
 * series to get intended motion, evolve D_k = M_k ∘ D_{k-1} ∘ M̂_k⁻¹
 * with a slow leak toward identity. Increments are in SOURCE px.
 */
export function buildResiduals(
  incs: FrameIncrement[],
  fps: number,
  smoothSec: number,
  leak = 0.005,
): Affine[] {
  const k = smoothSec * fps;
  const sth = gaussianSmoothArr(incs.map((i) => i.dtheta), k);
  const sls = gaussianSmoothArr(incs.map((i) => i.dlogs), k);
  const stx = gaussianSmoothArr(incs.map((i) => i.dtx), k);
  const sty = gaussianSmoothArr(incs.map((i) => i.dty), k);
  let D: Affine = IDENTITY;
  const out: Affine[] = [D];
  for (let i = 0; i < incs.length; i++) {
    const M = fromParams(incs[i].dtheta, Math.exp(incs[i].dlogs), incs[i].dtx, incs[i].dty);
    const Mhat = fromParams(sth[i], Math.exp(sls[i]), stx[i], sty[i]);
    D = compose(M, compose(D, invert(Mhat)));
    for (let j = 0; j < 6; j++) D[j] = (1 - leak) * D[j] + leak * IDENTITY[j];
    out.push([...D] as Affine);
  }
  return out;
}

/**
 * Cap the applied shake-correction translation per frame. During
 * violent maneuvers the residual reaches hundreds of px: full
 * correction keeps the WORLD stable but swings the rider (whom the
 * drone chases) outside any leash. caps = 80% of the tighter leash
 * slack for that frame's crop; above it the camera rides along.
 */
export function attenuateResiduals(Ds: Affine[], caps: number[], fps: number): Affine[] {
  const f = Ds.map((D, i) => {
    const mag = Math.hypot(D[2], D[5]);
    const cap = caps[Math.min(i, caps.length - 1)] ?? Infinity;
    return Math.min(1, cap / Math.max(mag, 1e-6));
  });
  const fs = gaussianSmoothArr(f, fps * 0.15);
  return Ds.map(
    (D, i) => [D[0], D[1], D[2] * fs[i], D[3], D[4], D[5] * fs[i]] as Affine,
  );
}

function gaussianSmoothArr(values: number[], sigma: number): number[] {
  if (sigma <= 0) return values.slice();
  const radius = Math.max(1, Math.ceil(sigma * 3));
  const kernel: number[] = [];
  let sum = 0;
  for (let i = -radius; i <= radius; i++) {
    const w = Math.exp(-(i * i) / (2 * sigma * sigma));
    kernel.push(w);
    sum += w;
  }
  return values.map((_, i) => {
    let acc = 0;
    for (let j = -radius; j <= radius; j++) {
      const idx = Math.min(values.length - 1, Math.max(0, i + j));
      acc += values[idx] * kernel[j + radius];
    }
    return acc / sum;
  });
}

// ---- FFT (radix-2, in-place) ----

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
