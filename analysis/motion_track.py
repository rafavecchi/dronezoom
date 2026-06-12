"""Motion-based rider tracking: align consecutive frames using measured
camera shift, diff them, find the moving blob near the prediction.

Validates against YOLO detections where available, then renders with the
fused track and measures output jitter + rider centering.
"""

import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from pipeline import (  # noqa: E402
    build_path,
    cam_pass,
    detect_pass,
    fill_gaps,
    gaussian_smooth,
    render,
)

clip = sys.argv[1]

samples = detect_pass(clip)
ts_s = np.array([s["t"] for s in samples])
yolo_cx = fill_gaps([s["box"]["cx"] if s["box"] else np.nan for s in samples])
yolo_cy = fill_gaps([s["box"]["cy"] if s["box"] else np.nan for s in samples])


def rider_at(t):
    return np.interp(t, ts_s, yolo_cx), np.interp(t, ts_s, yolo_cy)


cam = cam_pass(clip, rider_at)
W, H = int(cam["size"][0]), int(cam["size"][1])
fps = float(cam["fps"][0])

# ---- motion blob pass ----
AW, AH = 1024, 512
sx_, sy_ = W / AW, H / AH
cap = cv2.VideoCapture(clip)
prev = None
i = 0
# start prediction from the first YOLO box (the "seed click")
px, py = yolo_cx[0] / sx_, yolo_cy[0] / sy_  # analysis px
track_t, track_x, track_y, track_h, track_ok = [], [], [], [], []
SEARCH = 110  # analysis px search radius around prediction
vx = vy = 0.0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    t = i / fps
    g = cv2.cvtColor(cv2.resize(frame, (AW, AH)), cv2.COLOR_BGR2GRAY)
    if prev is not None:
        idx = min(max(i - 1, 0), len(cam["fx"]) - 1)
        dx, dy = cam["fx"][idx] / sx_, cam["fy"][idx] / sy_
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned = cv2.warpAffine(prev, M, (AW, AH))
        diff = cv2.absdiff(g, aligned)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        # ignore warp borders
        b = int(np.ceil(max(abs(dx), abs(dy)))) + 2
        diff[:b, :] = 0
        diff[-b:, :] = 0
        diff[:, :b] = 0
        diff[:, -b:] = 0
        # search window around prediction (with velocity)
        cx_pred = px + vx
        cy_pred = py + vy
        x0 = int(max(cx_pred - SEARCH, 0))
        x1 = int(min(cx_pred + SEARCH, AW))
        y0 = int(max(cy_pred - SEARCH, 0))
        y1 = int(min(cy_pred + SEARCH, AH))
        win = diff[y0:y1, x0:x1]
        found = False
        n = 0
        if win.size > 100:
            thr = max(10, 4 * np.median(diff[diff > 0]) if (diff > 0).any() else 10)
            mask = (win > thr).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
        if n > 1:
            # largest blob
            k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            if stats[k, cv2.CC_STAT_AREA] >= 12:
                bx, by = cents[k]
                nx, ny = x0 + bx, y0 + by
                vx, vy = 0.5 * vx + 0.5 * (nx - px), 0.5 * vy + 0.5 * (ny - py)
                px, py = nx, ny
                track_h.append(stats[k, cv2.CC_STAT_HEIGHT] * sy_)
                found = True
        if not found:
            px, py = px + vx, py + vy  # coast
            px = min(max(px, 0), AW)
            py = min(max(py, 0), AH)
            track_h.append(track_h[-1] if track_h else 80.0)
        track_t.append(t)
        track_x.append(px * sx_)
        track_y.append(py * sy_)
        track_ok.append(found)
    prev = g
    i += 1
cap.release()

track_t = np.array(track_t)
track_x = np.array(track_x)
track_y = np.array(track_y)
track_ok = np.array(track_ok)
print(f"motion track: {track_ok.mean() * 100:.0f}% frames with blob found ({track_ok.sum()}/{len(track_ok)})")

# validate against YOLO where YOLO was confident
errs = []
for s in samples:
    if s["box"] and s["box"]["score"] > 0.4:
        mx = np.interp(s["t"], track_t, track_x)
        my = np.interp(s["t"], track_t, track_y)
        errs.append(np.hypot(mx - s["box"]["cx"], my - s["box"]["cy"]))
if errs:
    print(f"vs confident YOLO ({len(errs)} pts): median err {np.median(errs):.0f}px, p90 {np.percentile(errs, 90):.0f}px")

# ---- build path from motion track (downsample to SAMPLE_INTERVAL grid) ----
hs = np.array(track_h, dtype=np.float64)
hs_smooth = gaussian_smooth(hs, fps)  # heavy: blob height is noisy
sample_ts = np.arange(0, track_t[-1], 0.2)
m_samples = [
    {
        "t": float(t),
        "box": {
            "cx": float(np.interp(t, track_t, track_x)),
            "cy": float(np.interp(t, track_t, track_y)),
            "w": 60.0,
            "h": float(np.clip(np.interp(t, track_t, hs_smooth) * 1.6, 60, 300)),
            "score": 1.0,
        },
    }
    for t in sample_ts
]

cum_x, cum_y = np.cumsum(cam["fx"]), np.cumsum(cam["fy"])
roll_cum = np.cumsum(cam["roll"])
cpath = build_path(m_samples, cam["ts"], cum_x, cum_y, W, H)
render(clip, "sim-v2-motion.mp4", cpath, cam["ts"], cum_x, cum_y, roll_cum)
print("rendered sim-v2-motion.mp4")

# rider centering in output: where does the track land in the output frame?
offs = []
for j, t in enumerate(track_t):
    camx = cum_x[min(j, len(cum_x) - 1)]
    camy = cum_y[min(j, len(cum_y) - 1)]
    pcx = np.interp(t, cpath["ts"], cpath["cx"]) + camx
    pcy = np.interp(t, cpath["ts"], cpath["cy"]) + camy
    cw = np.interp(t, cpath["ts"], cpath["cropW"])
    chh = np.interp(t, cpath["ts"], cpath["cropH"])
    pcx = min(max(pcx, cw / 2), W - cw / 2)
    pcy = min(max(pcy, chh / 2), H - chh / 2)
    offs.append((abs(track_x[j] - pcx) / (cw / 2), abs(track_y[j] - pcy) / (chh / 2)))
offs = np.array(offs)
print(f"rider offset from crop center (fraction of half-crop): x median {np.median(offs[:, 0]):.2f} p95 {np.percentile(offs[:, 0], 95):.2f}")
print(f"                                                       y median {np.median(offs[:, 1]):.2f} p95 {np.percentile(offs[:, 1], 95):.2f}")
