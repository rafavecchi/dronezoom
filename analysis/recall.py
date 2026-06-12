"""Why is detection recall low? Sweep thresholds, ROI sizes, model input."""

import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from pipeline import SAMPLE_INTERVAL, detect_persons, fill_gaps  # noqa: E402

clip = sys.argv[1]
cap = cv2.VideoCapture(clip)
fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

# Full-frame detection at every sample with a low threshold: what does
# raw recall look like, and what are the scores?
rows = []
t = 0.0
while t < dur:
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    if not ok:
        break
    dets = detect_persons(frame, thresh=0.10)
    best = dets[0] if dets else None
    rows.append((t, best["score"] if best else 0.0, best["h"] if best else 0.0))
    t += SAMPLE_INTERVAL
cap.release()

arr = np.array(rows)
scores = arr[:, 1]
print(f"samples: {len(arr)}")
for th in [0.10, 0.20, 0.25, 0.35, 0.50]:
    print(f"  full-frame recall @thresh {th}: {(scores >= th).mean() * 100:.0f}%")
hs = arr[arr[:, 1] > 0.1, 2]
print(f"rider h when detected: median {np.median(hs):.0f}px, p10 {np.percentile(hs, 10):.0f}, p90 {np.percentile(hs, 90):.0f}")
# score over time, coarse
for lo in range(0, int(dur), 5):
    seg = arr[(arr[:, 0] >= lo) & (arr[:, 0] < lo + 5)]
    print(f"  t={lo:3d}-{lo + 5:3d}s: mean best score {seg[:, 1].mean():.2f}")
