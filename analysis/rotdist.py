"""Distribution of per-frame fitted rotation in a time window, unclamped."""

import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from v3 import WIN, fit_similarity, grid_windows  # noqa: E402

clip, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
AW, AH = 1024, 512
cap = cv2.VideoCapture(clip)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(t0 * fps))
win = cv2.createHanningWindow((WIN, WIN), cv2.CV_32F)
prev = None
ths, ts_, inl = [], [], []
for i in range(int((t1 - t0) * fps)):
    ok, frame = cap.read()
    if not ok:
        break
    g = cv2.cvtColor(cv2.resize(frame, (AW, AH)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    if prev is not None:
        centers, shifts, weights = [], [], []
        for x0, y0 in grid_windows():
            (dx, dy), pk = cv2.phaseCorrelate(
                prev[y0 : y0 + WIN, x0 : x0 + WIN], g[y0 : y0 + WIN, x0 : x0 + WIN], win
            )
            if pk > 0.03 and abs(dx) < WIN * 0.3 and abs(dy) < WIN * 0.3:
                centers.append((x0 + WIN / 2, y0 + WIN / 2))
                shifts.append((dx, dy))
                weights.append(pk)
        if len(centers) >= 3:
            M = fit_similarity(centers, shifts, weights)
            th = np.arctan2(M[1, 0], M[0, 0])
            # inliers
            n_in = 0
            for c, s in zip(centers, shifts):
                pred = M @ np.array([c[0], c[1], 1.0])
                if np.hypot(pred[0] - (c[0] + s[0]), pred[1] - (c[1] + s[1])) < 3:
                    n_in += 1
            ths.append(np.degrees(th))
            inl.append(n_in)
            ts_.append(t0 + i / fps)
    prev = g
cap.release()

ths = np.array(ths)
inl = np.array(inl)
print(f"window {t0}-{t1}s: {len(ths)} pairs, windows used {inl.mean():.1f} avg")
print(f"|dtheta|/frame deg: median {np.median(np.abs(ths)):.2f}, p90 {np.percentile(np.abs(ths), 90):.2f}, max {np.abs(ths).max():.2f}")
print(f"frames over 2.0 deg clamp: {(np.abs(ths) > 2.0).mean() * 100:.0f}%")
print(f"frames over 2.0 deg with >=4 inliers (trustworthy but clamped): {((np.abs(ths) > 2.0) & (inl >= 4)).mean() * 100:.0f}%")
