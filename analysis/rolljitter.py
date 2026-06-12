"""Rotational (roll) jitter over time: left/right half-frame vertical
differential per frame pair, high-passed. Reports per-5s segments."""

import sys

import cv2
import numpy as np

path = sys.argv[1]
cap = cv2.VideoCapture(path)
fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
AW, AH = 1024, 512
win = cv2.createHanningWindow((AW // 2, AH), cv2.CV_32F)
prev = None
rolls = []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    g = cv2.cvtColor(cv2.resize(frame, (AW, AH)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    if prev is not None:
        (_, ldy), _ = cv2.phaseCorrelate(prev[:, : AW // 2], g[:, : AW // 2], win)
        (_, rdy), _ = cv2.phaseCorrelate(prev[:, AW // 2 :], g[:, AW // 2 :], win)
        roll = ((rdy - ldy) * (H / AH)) / (W / 2)
        rolls.append(roll if abs(roll) < 0.05 else 0.0)
    prev = g
cap.release()

r = np.degrees(np.cumsum(rolls))
k = max(3, int(fps * 0.5) | 1)
pad = k // 2
sm = np.convolve(np.pad(r, pad, mode="edge"), np.ones(k) / k, mode="valid")
hf = r - sm
print(f"{path}: roll jitter RMS {hf.std():.3f} deg, p95 {np.percentile(np.abs(hf), 95):.3f} deg")
for lo in range(0, int(len(hf) / fps), 5):
    seg = hf[int(lo * fps) : int((lo + 5) * fps)]
    if len(seg):
        print(f"  t={lo:3d}-{lo + 5:3d}s: rms {seg.std():.3f} deg")
