"""At each time: show ALL YOLO person detections (so we can see the
group), the blob-track position, and the prior track position — to see
what pulls the lock onto the wrong people."""

import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from pipeline import detect_persons  # noqa: E402
from v3 import pass_a  # noqa: E402
from v4 import track_blobs  # noqa: E402
from pipeline import detect_pass  # noqa: E402

clip = sys.argv[1]
t_lo, t_hi = float(sys.argv[2]), float(sys.argv[3])
A = pass_a(clip)
fps = float(A["fps"][0])
W, H = int(A["size"][0]), int(A["size"][1])
yolo = detect_pass(clip)
track = track_blobs(clip, A, yolo)

cap = cv2.VideoCapture(clip)
t = t_lo
while t <= t_hi:
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    if not ok:
        break
    dets = detect_persons(frame, thresh=0.25)
    j = min(int(t * fps), len(track["ts"]) - 1)
    bx, by = track["x"][j], track["y"][j]
    descr = " | ".join(
        f"({d['cx']:.0f},{d['cy']:.0f} s{d['score']:.2f})" for d in sorted(dets, key=lambda z: -z["score"])[:6]
    )
    print(f"t={t:4.1f}  blob=({bx:.0f},{by:.0f})  {len(dets)} ppl: {descr}")
    t += 0.5
cap.release()
