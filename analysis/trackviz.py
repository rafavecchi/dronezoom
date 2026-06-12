"""Draw the blob track position onto source frames at given times."""

import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from pipeline import detect_pass  # noqa: E402
from v3 import pass_a  # noqa: E402
from v4 import track_blobs  # noqa: E402

clip = sys.argv[1]
times = [float(x) for x in sys.argv[2:]]
A = pass_a(clip)
fps = float(A["fps"][0])
yolo = detect_pass(clip)
track = track_blobs(clip, A, yolo)

cap = cv2.VideoCapture(clip)
for t in times:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, frame = cap.read()
    if not ok:
        continue
    x = np.interp(t, track["ts"], track["x"])
    y = np.interp(t, track["ts"], track["y"])
    cv2.circle(frame, (int(x), int(y)), 40, (0, 255, 255), 4)
    cv2.imwrite(f"tv_{t:.0f}.jpg", cv2.resize(frame, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 88])
cap.release()
print("done")
