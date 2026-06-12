"""Calibrate cv2.phaseCorrelate sign + alignment quality on real frames."""

import sys

import cv2
import numpy as np

clip = sys.argv[1]
cap = cv2.VideoCapture(clip)
cap.set(cv2.CAP_PROP_POS_FRAMES, 450)
ok, f1 = cap.read()
ok, f2 = cap.read()
cap.release()

AW, AH = 1024, 512
g1 = cv2.cvtColor(cv2.resize(f1, (AW, AH)), cv2.COLOR_BGR2GRAY).astype(np.float32)
g2 = cv2.cvtColor(cv2.resize(f2, (AW, AH)), cv2.COLOR_BGR2GRAY).astype(np.float32)
win = cv2.createHanningWindow((AW, AH), cv2.CV_32F)
(dx, dy), pk = cv2.phaseCorrelate(g1, g2, win)
print(f"phaseCorrelate(g1,g2) = ({dx:.2f},{dy:.2f}) peak {pk:.3f}")

base = cv2.absdiff(g2, g1).mean()
for s in (+1, -1):
    M = np.float32([[1, 0, s * dx], [0, 1, s * dy]])
    aligned = cv2.warpAffine(g1, M, (AW, AH))
    m = int(np.ceil(abs(dx)) + np.ceil(abs(dy))) + 2
    d = cv2.absdiff(g2, aligned)[m:-m, m:-m].mean()
    print(f"align prev by {s:+d}*d: residual {d:.2f} (unaligned {base:.2f})")

# synthetic ground truth: roll g1 by known amount
gs = np.roll(np.roll(g1, 7, axis=1), -4, axis=0)  # content moves +7x, -4y
(sdx, sdy), _ = cv2.phaseCorrelate(g1, gs, win)
print(f"known content motion (+7,-4) -> phaseCorrelate returns ({sdx:.2f},{sdy:.2f})")
