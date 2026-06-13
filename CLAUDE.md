# DroneZoom — working guide for future sessions

Browser + PC app that auto-zooms onto Rafael riding MTB in DJI drone
footage and stabilizes it. **Read this before changing tracking/framing
code — it records what works, what failed, and how to debug, so we don't
re-derive it.**

## The #1 process rule (learned the hard way)

Offline metrics (jitter RMS, track-vs-YOLO error) do **NOT** reliably
predict what Rafael perceives. I once burned ~8 rounds tuning smoothness
blind. **Before claiming a fix: render the clip, extract frames at the
complained-about timestamps, LOOK at them, and get Rafael's visual
confirmation.** `analysis/annotate.py` (draws track=yellow, crop=green,
YOLO=red on source frames) is the single most valuable tool here — it's
how we discovered "focuses on the group" was a wrong-target lock and
"out of frame at s25" was crop-lag, not a tracking failure.

When Rafael reports a bad moment, the first question is always: **is the
tracker on him (→ framing/path problem) or off him (→ tracking problem)?**
Answer it with annotate.py before touching code.

## Two delivery paths (keep them in sync)

1. **PC worker** (PRIMARY — what Rafael uses from his phone). He shares a
   clip to Google Drive `DroneZoom/inbox`; `worker/worker.py` polls it,
   runs the Python pipeline, uploads the result to `DroneZoom/done`.
   - The worker imports `analysis/pipeline.py` + `analysis/v4.py` + `v3.py`
     **LIVE from disk** — never leave experiments uncommitted in those
     files or the next phone job uses them.
   - Autostart: Startup-folder VBS `DroneZoomWorker.vbs` (hidden).
     Single-instance socket lock on 127.0.0.1:47821. Log: `worker/worker.log`.
   - OAuth: Desktop client in Google Cloud project 9652544572,
     `worker/token.json` (gitignored).
2. **Web app** (`src/*.ts`, deployed to GitHub Pages) — manual/tuning UI.
   Separate code path: `analyze.ts` (tracking) + `path.ts` + `motion.ts`.
   When you change pipeline behavior, port it here too.

After changing worker Python: commit, restart the worker (kill pythonw +
relaunch), and optionally upload a sample render. After changing TS:
`npm run build`, commit, push (auto-deploys to Pages).

## Pipeline (per clip)

1. **Camera motion** (`v3.pass_a` / `motion.ts`): grid phase-correlation →
   per-frame **similarity** transform (translation + rotation + zoom).
   Top-down drone yaw is REAL image rotation (>180°/clip); altitude = real
   zoom. Translation-only models fail. Sanity-clamp per-frame fits, but
   trust high-consensus (≥4 inlier windows) fits to larger angles.
2. **Shake residual** `D_k` (`residuals_incremental`): split each frame's
   transform into intended (smoothed) + shake; `D` = the shake part,
   leaked toward identity. Renderer counter-rotates/scales by `D`.
3. **Rider tracking** (`track_blobs` / `analyze.ts`): YOLO can't see a
   top-down rider (~15% recall), so the rider is a **motion blob** in the
   shake-aligned frame difference (center-surround filter). YOLO only
   **seeds** identity and **re-anchors** every 0.2s.
4. **Camera path** (`build_path_v4` / `path.ts`): rider track → smoothed,
   leashed crop path. Render = same path for preview and export.

## Validated config / defaults (don't regress these)

- Framing 5× rider height, 1s smoothing, max 4× zoom, leash 0.45 H / 0.35 V.
- Background shake target ≈ **3.6–4.7px** (0.15s high-pass RMS). The 11px
  version was the cap+Hampel regression — never reintroduce it.
- Detection **size gate**: reject detections <0.5× / >2× the running-median
  rider height. THIS stops the lock jumping to a distant group of people.
- Blob **jump gate**: reject candidates >120 analysis-px (≈56 BW-px in TS)
  from the velocity prediction. Rejects noise-grabs; 40px was too tight
  (broke whip recovery).
- YOLO **rescue-snap threshold 0.38** (detections are size-gated, so trust
  low scores) — fixes low-contrast drift where the rider scores 0.40–0.44.
- Leash reference = **radius-1 median** of the raw track (NOT heavier
  median/hampel — those clip fast switchbacks and lag the crop).
- Frame-boundary clamp in **FRAME coords** (shift bounds by the smoothed
  residual translation), not world coords — else big maneuvers push the
  rider out of frame near edges.
- **Edge zoom-out**: widen up to 2× as the rider's tracked distance to the
  nearest source-frame edge shrinks (ramps over the outer 22%). Keeps the
  rider framed when cornered. Trigger is the rider's position vs the frame
  — reliable.
- **Zoom-out-on-pan**: widen up to 1.8× during violent camera pans
  (rolling-max speed envelope).

## Failed experiments — DO NOT repeat (all measured)

- **Correction cap** (attenuate_residuals): scaled the whole shake
  correction → 6× jitter regression. The user's "very shaky". Reverted.
- **Hampel leash reference + extra gaussian**: part of the same shake
  regression; also clips fast moves. Reverted.
- **Appearance gate** (HSV / H-S histogram) on rescue snaps: blocks
  legitimate re-acquisition when the rider enters shadow (color changes).
  track p90 139→568px. Reverted.
- **Physics/jump gate at 40px**: too tight, broke whip-pan re-acquisition
  (clip2 found 99→84%). Use 120px.
- **Lost-subject auto zoom-out** (trigger = YOLO silence): fires 82% of
  top-down clips (YOLO is sparse everywhere; the blob tracks fine through
  it). Would zoom-pump the whole video. No reliable online "lost" signal
  exists — during a real blackout the blob confidently locks onto grass
  (quality HIGHER than normal). Use the **edge** trigger instead.
- **CSRT as primary tracker**: on a tiny top-down rider it drifts the same
  way the blob does (opencv-contrib installed; `analysis/v5.py` has a
  CSRT+L1 prototype if revisited). The motion blob + gates is better here.

## Known remaining limits

- **Detection blackout** (e.g. s24-26 of `video~3`): rider invisible to
  YOLO even full-frame (deep shadow + frame corner). The blob still
  tracks via motion, and the frame-coord clamp + edge-zoom now keep the
  rider in-frame at the corner — but at the extreme edge they sit near the
  output border. A learned re-ID model would help marginally; not worth it.
- **Crop-lag on the sharpest hairpins**: mostly fixed by the frame-coord
  clamp + radius-1 leash; a residual fraction-of-a-second lag can remain.

## Debugging from my (Claude's) side

- `worker/fetch.py` — pull Rafael's clips from Drive (he can't easily
  screen-record). Lists DroneZoom folders; downloads by file id.
- `analysis/annotate.py <clip> <t1> <t2> ...` — track+crop+YOLO overlay.
- `analysis/whoswho.py <clip> <t0> <t1>` — all detections + blob per frame.
- `analysis/centering.py` / shake snippet — output-side metrics.
- The shake metric: phase-correlate consecutive OUTPUT frames, cumulative
  path, 0.15s high-pass, RMS. The "subject-bounce" metric (rider position
  WITHIN the output frame, high-passed) matches "shaky" better when the
  subject is the focus.
- `analysis/cache/*.npz` caches the per-clip camera + detection passes —
  delete after changing those passes.

## Test clips

`src3.mp4` (= `dji_fly_…video~3`, high-altitude bike park, sparse YOLO,
the s24-26 blackout) and `video~2.mp4` (lower, faster). Both gitignored.
Always regression-check BOTH before shipping a tracking/path change.
