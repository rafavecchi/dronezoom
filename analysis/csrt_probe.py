"""Can CSRT (appearance tracker) hold the rider through the s24-26
detection blackout where the motion blob drifts to grass? Seed it on the
rider before the blackout and trace it. Ground truth: rider re-emerges
at ~(1814,689) at t26 having ridden RIGHT along the trail."""

import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from pipeline import detect_persons  # noqa: E402

clip = r"..\src3.mp4"
cap = cv2.VideoCapture(clip)
fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Seed on the rider at t=22.0 (well before the blackout)
seed_t = 22.0
cap.set(cv2.CAP_PROP_POS_FRAMES, int(seed_t * fps))
ok, frame = cap.read()
dets = sorted(detect_persons(frame, thresh=0.25), key=lambda d: -d["score"])
if not dets:
    print("no seed detection at t=22 — trying t=21.5")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(21.5 * fps))
    ok, frame = cap.read()
    dets = sorted(detect_persons(frame, thresh=0.25), key=lambda d: -d["score"])
b = dets[0]
print(f"seed @ t{seed_t}: ({b['cx']:.0f},{b['cy']:.0f}) h{b['h']:.0f} s{b['score']:.2f}")
roi = (
    int(b["cx"] - b["w"] / 2),
    int(b["cy"] - b["h"] / 2),
    int(b["w"]),
    int(b["h"]),
)
tracker = cv2.TrackerCSRT_create()
tracker.init(frame, roi)

# Run forward through the blackout, printing position every ~0.3s
i = int(seed_t * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, i)
print("\nt     CSRT pos        conf(box stable?)")
last_print = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    t = i / fps
    got, box = tracker.update(frame)
    if t - last_print >= 0.3:
        if got:
            x, y, w, h = box
            print(f"{t:.1f}  ({x + w / 2:.0f},{y + h / 2:.0f}) wh({w:.0f}x{h:.0f})  ok")
        else:
            print(f"{t:.1f}  LOST (update returned False)")
        last_print = t
    if t > 28:
        break
    i += 1
cap.release()
print("\nground truth: rider rides RIGHT, re-emerges ~(1814,689) at t26")
