"""Is a 'lost me' moment a TRACK failure or a PATH lag? Compute, per
0.1s, the rider-track offset from the rendered crop center (fraction of
half-crop), list sustained excursions, and compare track vs YOLO there."""

import sys

import numpy as np

sys.path.insert(0, ".")
from pipeline import detect_pass, gaussian_smooth  # noqa: E402
from v3 import apply_a, pass_a  # noqa: E402
from v4 import (  # noqa: E402
    attenuate_residuals,
    build_path_v4,
    residuals_incremental,
    to_source_Ms,
    track_blobs,
)

clip = sys.argv[1]
A = pass_a(clip)
fps = float(A["fps"][0])
W, H = int(A["size"][0]), int(A["size"][1])
Ms_src = to_source_Ms(A["Ms"], W, H)
Ds = residuals_incremental(Ms_src, fps)
yolo = detect_pass(clip)
track = track_blobs(clip, A, yolo)

speeds = np.hypot(Ms_src[:, 0, 2], Ms_src[:, 1, 2])
r = int(fps * 0.3)
env = np.array([speeds[max(0, i - r) : i + r + 1].max() for i in range(len(speeds))])
sp_s = gaussian_smooth(env, fps * 0.3)
x01 = np.clip((sp_s - 0.021 * W) / (0.052 * W - 0.021 * W), 0, 1)
wfac = 1 + 0.8 * (x01 * x01 * (3 - 2 * x01))
widen = (np.arange(len(wfac)) / fps, wfac)
path, Ds = build_path_v4(track, Ds, fps, W, H, widen=widen)

ts = track["ts"]
offs = np.zeros((len(ts), 2))
for j, t in enumerate(ts):
    D = Ds[min(j + 1, len(Ds) - 1)]
    sxp = np.interp(t, path["ts"], path["cx"])
    syp = np.interp(t, path["ts"], path["cy"])
    cw = np.interp(t, path["ts"], path["cropW"])
    qx, qy = apply_a(D, sxp, syp)
    qx = min(max(qx, cw / 2), W - cw / 2)
    ch_ = cw * H / W
    qy = min(max(qy, ch_ / 2), H - ch_ / 2)
    offs[j] = (track["x"][j] - qx) / (cw / 2), (track["y"][j] - qy) / (ch_ / 2)

mag = np.hypot(offs[:, 0], offs[:, 1])
print(f"offs: med {np.median(mag):.2f} p90 {np.percentile(mag, 90):.2f} p99 {np.percentile(mag, 99):.2f}")
# sustained excursions: |off| > 0.85 for > 0.3s
bad = mag > 0.85
events = []
start = None
for j, b in enumerate(bad):
    if b and start is None:
        start = j
    if not b and start is not None:
        if ts[j] - ts[start] > 0.3:
            events.append((ts[start], ts[j], mag[start:j].max()))
        start = None
if start is not None:
    events.append((ts[start], ts[-1], mag[start:].max()))
yts = np.array([s["t"] for s in yolo])
for t0, t1, peak in events:
    # was the TRACK right there? compare to nearest confident YOLO
    errs = [
        np.hypot(
            np.interp(s["t"], ts, track["x"]) - s["box"]["cx"],
            np.interp(s["t"], ts, track["y"]) - s["box"]["cy"],
        )
        for s in yolo
        if s["box"] and s["box"]["score"] > 0.35 and t0 - 1 <= s["t"] <= t1 + 1
    ]
    verdict = f"track-vs-YOLO {np.median(errs):.0f}px" if errs else "no YOLO nearby"
    print(f"  excursion {t0:5.1f}-{t1:5.1f}s  peak {peak:.2f}  {verdict}")
