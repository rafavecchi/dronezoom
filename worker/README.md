# DroneZoom PC worker

Watches `Google Drive/DroneZoom/inbox` and turns every clip dropped
there into a stabilized, auto-zoomed `*-dronezoom.mp4` in
`DroneZoom/done`. Sources are filed into `processed/` (or `failed/`
with an error note). Runs the validated Python pipeline (analysis/).

## Phone workflow

Share a clip from DJI Fly / gallery → Google Drive app → save into
`DroneZoom/inbox`. A few minutes later the result appears in
`DroneZoom/done`. The folders are created automatically.

## One-time setup

1. Google Cloud console (same project as the web app) → APIs & Services
   → Credentials → Create credentials → OAuth client ID → **Desktop
   app**. Download the JSON and save it as `worker/client_secret.json`.
2. Run `worker/run-worker.bat` once at the PC — a browser consent opens;
   approve it. The token persists in `worker/token.json`.
3. The worker is registered as a logon task ("DroneZoomWorker"), so it
   restarts with Windows. Log: `worker/worker.log`.

## Notes

- Seeding is automatic: the first confident person detection in the clip
  anchors the track (no seed click). If no rider is ever detected the
  job lands in `failed/` with an explanation.
- Defaults: framing 5x rider height, 1s smoothing, max 4x zoom,
  zoom-out-on-pan enabled, 1080p H.264 output.
