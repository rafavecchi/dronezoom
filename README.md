# DroneZoom

Browser app that auto-zooms and smooths drone footage of a mountain biker.
Everything runs client-side — no uploads.

## How it works (milestone 1)

1. Load an MP4 from your DJI Mini 2 and **click the rider** in the first frame
   to seed the tracker.
2. **Analyze** seeks through the clip (default every 0.2s), runs YOLO11n person
   detection via `onnxruntime-web` (WebGPU, WASM fallback), and follows the
   seeded rider with a constant-velocity, distance-gated tracker — a detection
   outside the gate counts as a miss, never a target switch. When the full
   frame misses, detection re-runs on a zoomed window around the predicted
   position (recovers riders only ~20px tall in downscaled 4K).
3. The raw track is gap-filled, median-filtered (kills single-frame outliers),
   and Gaussian-smoothed into a cinematic crop path (heavier smoothing on zoom
   level to avoid "breathing").
4. Live preview renders the cropped/zoomed view next to the original with
   tracking overlays. Framing tightness and smoothness sliders re-shape the
   path instantly without re-running detection.

## Run

```sh
npm install
npm run dev
```

Open the printed URL in Chrome/Edge (WebGPU). Firefox/Safari fall back to CPU
inference — works, but the analyze pass is much slower.

## Export to Google Drive

The "→ Drive" button uploads the last export to your Drive (scope
`drive.file`: the app can only see files it creates). One-time setup:

1. [console.cloud.google.com](https://console.cloud.google.com) → create
   (or pick) a project.
2. APIs & Services → Library → enable **Google Drive API**.
3. APIs & Services → OAuth consent screen → External → add yourself as a
   test user.
4. APIs & Services → Credentials → Create credentials → **OAuth client
   ID** → Web application → add `http://localhost:5173` under
   *Authorized JavaScript origins*.
5. Copy the client ID; the app prompts for it on first upload (stored in
   localStorage).

## Roadmap

- **Milestone 2 — Export:** WebCodecs decode → render crop per frame → encode →
  mp4box.js mux. Full-resolution output of the preview.
- **Milestone 3 — Editing:** trim/segments, per-segment framing override,
  manual crop keyframes where detection fails (trees, shadows).
- **Milestone 4 — Stabilization pass:** optional vid.stab via a custom
  ffmpeg.wasm build for residual jitter at high zoom.

## Notes

- `public/models/yolo11n-detect.onnx` is an Ultralytics YOLO11 export —
  **AGPL-3.0** (see `public/models/` license). Fine for personal use; swap for
  an Apache-2.0 detector (e.g. YOLOX) before commercializing.
- `onnxruntime-web` version is pinned; the WASM CDN path in `src/detect.ts`
  must match it.
