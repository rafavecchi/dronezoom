# DroneZoom

Browser app that auto-zooms and smooths drone footage of a mountain biker.
Everything runs client-side — no uploads.

## How it works (milestone 1)

1. Load an MP4 from your DJI Mini 2.
2. **Analyze** seeks through the clip (default every 0.2s), runs YOLO11n person
   detection via `onnxruntime-web` (WebGPU, WASM fallback), and tracks the rider
   across samples.
3. The raw track is gap-filled and Gaussian-smoothed into a cinematic crop path
   (heavier smoothing on zoom level to avoid "breathing").
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
