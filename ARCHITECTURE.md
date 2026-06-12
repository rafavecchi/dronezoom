# DroneZoom v2 architecture (validated in analysis/v4.py)

Validated offline against `dji_fly_20260611_180440_512_*.mp4` (top-down
follow shot, 1080p): blob track on 98% of frames, rider median offset
0.23/0.27 of half-crop, output visually smooth outside whip-pans.

## Key findings from the test clip

- YOLO person detection mostly FAILS on top-down drone footage (15%
  recall @0.35) — a person+bike from above doesn't look like a person.
  YOLO is only used to seed identity and occasionally re-anchor.
- The rider is found per-frame as a **motion blob**: align consecutive
  frames with the measured camera transform, diff, center-surround
  filter (compact blobs pop, parallax bands cancel), pick peak near
  prediction. Works at every frame, any viewpoint.
- Camera motion needs a **similarity model** (translation + rotation +
  scale): top-down + drone yaw = real image rotation (this clip spans
  >180° of yaw); altitude change = real zoom (5x over the clip).
  Translation-only stabilization can't represent this.
- rVFC-based measurement drops frames under load; **seek-stepping every
  frame** is mandatory for correction coverage (a dropped frame is an
  uncorrected frame).

## Single analysis pass (per frame k)

1. Grab grayscale at 1024x512 (GW×GH).
2. Grid phase correlation: 6 windows of 256², fit similarity transform
   M_k (frame k-1 → k content motion) with per-window peak weights,
   outlier rejection (residual > 3px), and sanity clamp (|dθ|>2°,
   |ds-1|>3%, |t|>200px → median-translation fallback).
3. Aligned diff: warp prev by M_k, absdiff. Adaptive quality gate:
   mean residual > max(2.2×running-median, 25) = whip pan → coast.
4. Blob track: response = blur(diff,σ3) − 0.8·blur(diff,σ12), peak of
   response × gaussian-proximity(prediction, σ=80 analysis px), accept
   if response > 4; centroid refine ±12px; velocity EMA 0.6/0.4; height
   from diff column extent. Coast with decaying velocity on miss.
5. Every ~0.2s: YOLO on ROI around prediction (tracker.ts identity
   gating as today); if score>0.3 and within 100 analysis px, blend
   blob position 0.7/0.3 toward YOLO. Seeded by the user's click.

## After the pass

- Smooth per-frame increments (dθ, dlog s, dtx, dty) with σ=smooth_sec;
  rebuild intended motion M̂_k; shake residual evolves as
  D_k = M_k ∘ D_{k-1} ∘ M̂_k⁻¹, leaked toward identity (0.005/frame).
  D stays near-identity — no global-frame blowup despite unbounded yaw.
- Rider path: r̂_k = D_k⁻¹(blob_k) ("intended" coords). Downsample to
  0.2s grid, median filter, Gaussian+leash+boundary-constrain loop as
  today (leash 0.45/0.35, margins 2%/2.5%).
- Render frame k: q = D_k(S(t)), clamp to frame; counter-rotate by
  D's θ_k about q; scale outW/(cropW·s_k). Same renderView for preview
  and export.

## Jitter measurement (analysis/analyze.py)

Phase-correlate consecutive output frames, cumulative path, high-pass
(0.5s), RMS. Source floor ≈ x30/y9 px. Note: a tight follow-cam has
real intentional motion in this metric; compare like-for-like paths.

## Open items

- Whip-pan moments: rider hits source frame edge; output can only pin
  him at crop edge (physics). Could zoom out during high |D| moments.
- Browser media pipeline in the automated Chrome profile stalls on all
  video (networkState=2 forever) — test via analysis/*.py instead.
- analysis/cache/*.npz caches detection + transform passes per clip.
