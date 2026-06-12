import type { Detection, Region } from './detect';
import {
  appearanceSimilarity,
  blendAppearance,
  captureAppearance,
  type Appearance,
} from './appearance';

export interface Point {
  x: number;
  y: number;
}

const MAX_PREDICT_SEC = 1.5;
const STALE_SEC = 2.5;

/**
 * Single-target tracker seeded by a user click. Identity is held by two
 * signals working together:
 *
 * - Motion: constant-velocity prediction with a speed-limited gate, so a
 *   detection across the frame can never be matched between two samples.
 * - Appearance: a color histogram captured from the seeded rider. Every
 *   candidate must resemble the template, and the longer the track has
 *   been lost, the stricter the resemblance required — wrong-person
 *   re-acquisition was exactly the failure mode this prevents.
 *
 * Works backward in time too (negative dt), so the caller can track both
 * directions from the seeded frame.
 */
export class Tracker {
  private last: Detection | null = null;
  private lastT = 0;
  private vx = 0;
  private vy = 0;
  private template: Appearance | null;
  private readonly diag: number;

  constructor(
    private seed: Point,
    private srcW: number,
    private srcH: number,
    template: Appearance | null = null,
  ) {
    this.diag = Math.hypot(srcW, srcH);
    this.template = template;
  }

  /** Share the learned appearance with a second pass (e.g. backward). */
  get appearance(): Appearance | null {
    return this.template;
  }

  predict(t: number): Point {
    if (!this.last) return this.seed;
    const dt = clamp(t - this.lastT, -MAX_PREDICT_SEC, MAX_PREDICT_SEC);
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
      : Math.max(480, this.srcH / 3);
    return {
      x: clamp(p.x - size / 2, 0, Math.max(0, this.srcW - size)),
      y: clamp(p.y - size / 2, 0, Math.max(0, this.srcH - size)),
      w: Math.min(size, this.srcW),
      h: Math.min(size, this.srcH),
    };
  }

  match(frame: CanvasImageSource, dets: Detection[], t: number): Detection | null {
    const p = this.predict(t);
    const dtAbs = this.last ? Math.abs(t - this.lastT) : 0;
    const stale = this.last !== null && dtAbs > STALE_SEC;
    const gate = this.gateRadius(t);
    // The longer we've been lost, the more the candidate must look like
    // the rider before we'll lock back on.
    const simFloor = !this.template
      ? 0
      : !this.last
        ? 0.3
        : stale
          ? 0.5
          : dtAbs > 0.6
            ? 0.4
            : 0.25;

    let best: Detection | null = null;
    let bestApp: Appearance | null = null;
    let bestCost = Infinity;
    for (const d of dets) {
      const dist = Math.hypot(d.cx - p.x, d.cy - p.y);
      if (dist > gate) continue;
      if (this.last) {
        const ratio = d.h / this.last.h;
        const [lo, hi] = stale ? [0.3, 3.0] : [0.4, 2.5];
        if (ratio < lo || ratio > hi) continue;
      }
      const app = captureAppearance(frame, d);
      const sim = this.template ? appearanceSimilarity(this.template, app) : 0.5;
      if (sim < simFloor) continue;
      const cost = 0.6 * (dist / gate) + (1 - sim) - 0.1 * d.score;
      if (cost < bestCost) {
        bestCost = cost;
        best = d;
        bestApp = app;
      }
    }

    if (best) {
      const dt = t - this.lastT;
      if (this.last && !stale && Math.abs(dt) > 1e-6 && Math.abs(dt) < MAX_PREDICT_SEC) {
        this.vx = 0.5 * this.vx + (0.5 * (best.cx - this.last.cx)) / dt;
        this.vy = 0.5 * this.vy + (0.5 * (best.cy - this.last.cy)) / dt;
        const speed = Math.hypot(this.vx, this.vy);
        const maxSpeed = 0.5 * this.diag;
        if (speed > maxSpeed) {
          this.vx *= maxSpeed / speed;
          this.vy *= maxSpeed / speed;
        }
      } else {
        // Fresh acquisition or stale re-acquire: old velocity is noise.
        this.vx = 0;
        this.vy = 0;
      }
      if (!this.template) {
        this.template = bestApp;
      } else if (bestApp && dtAbs <= 0.6) {
        blendAppearance(this.template, bestApp, 0.05);
      }
      this.last = best;
      this.lastT = t;
    }
    return best;
  }

  private gateRadius(t: number): number {
    if (!this.last) return Math.max(160, 0.07 * this.diag); // around the seed click
    const dtAbs = Math.abs(t - this.lastT);
    // Long loss: search wide, but match() demands a strong appearance hit.
    if (dtAbs > STALE_SEC) return 0.5 * this.diag;
    const maxSpeed = 0.4 * this.diag; // px/s — generous for a fast descent
    return Math.min(0.3 * this.diag, 0.04 * this.diag + maxSpeed * dtAbs);
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}
