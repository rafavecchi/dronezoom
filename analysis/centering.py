"""Where is the rider in the export frame over time? (YOLO works on the
zoomed output — rider is big there.) Reports offset from center."""

import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from pipeline import detect_persons  # noqa: E402

clip = sys.argv[1]
cap = cv2.VideoCapture(clip)
fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

t = 0.0
rows = []
while t < dur:
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    if not ok:
        break
    dets = detect_persons(frame, thresh=0.25)
    if dets:
        d = dets[0]
        rows.append((t, (d["cx"] - W / 2) / (W / 2), (d["cy"] - H / 2) / (H / 2), d["score"]))
    else:
        rows.append((t, np.nan, np.nan, 0.0))
    t += 0.5
cap.release()

arr = np.array(rows)
det = ~np.isnan(arr[:, 1])
print(f"rider detected in export: {det.mean() * 100:.0f}% of {len(arr)} samples")
print("offset from center (fraction of half-frame, + = right/down):")
for lo in range(0, int(dur), 5):
    seg = arr[(arr[:, 0] >= lo) & (arr[:, 0] < lo + 5)]
    d2 = ~np.isnan(seg[:, 1])
    if d2.any():
        print(
            f"  t={lo:3d}-{lo + 5:3d}s: found {d2.mean() * 100:3.0f}%  x {np.nanmedian(seg[:, 1]):+.2f}  y {np.nanmedian(seg[:, 2]):+.2f}"
        )
    else:
        print(f"  t={lo:3d}-{lo + 5:3d}s: found   0%  -- rider not visible/detected --")
