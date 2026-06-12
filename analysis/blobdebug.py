"""Visualize aligned-diff blobs + track vs YOLO at chosen times."""

import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from pipeline import detect_pass, fill_gaps  # noqa: E402
from v3 import AW, AH, pass_a  # noqa: E402

clip = sys.argv[1]
times = [float(x) for x in sys.argv[2:]] or [3.0, 12.0, 25.0, 38.0]

A = pass_a(clip)
Ms = A["Ms"]
fps = float(A["fps"][0])
W, H = int(A["size"][0]), int(A["size"][1])

yolo = detect_pass(clip)
yts = np.array([s["t"] for s in yolo])
ycx = fill_gaps([s["box"]["cx"] if s["box"] else np.nan for s in yolo])
ycy = fill_gaps([s["box"]["cy"] if s["box"] else np.nan for s in yolo])

cap = cv2.VideoCapture(clip)
for t in times:
    i = int(t * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, i - 1)
    ok, f1 = cap.read()
    ok, f2 = cap.read()
    if not ok:
        continue
    g1 = cv2.cvtColor(cv2.resize(f1, (AW, AH)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(cv2.resize(f2, (AW, AH)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    M = Ms[min(i - 1, len(Ms) - 1)]
    aligned = cv2.warpAffine(g1, M, (AW, AH))
    diff = cv2.GaussianBlur(cv2.absdiff(g2, aligned), (5, 5), 0)
    print(f"t={t}: mean residual {diff[8:-8, 8:-8].mean():.2f}, max {diff.max():.0f}")
    vis = cv2.applyColorMap(np.clip(diff * 6, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    # overlay YOLO interp position
    yx, yy = np.interp(t, yts, ycx) * AW / W, np.interp(t, yts, ycy) * AH / H
    cv2.circle(vis, (int(yx), int(yy)), 12, (255, 255, 255), 2)
    cv2.imwrite(f"diff_t{t:.0f}.jpg", vis)
cap.release()
print("wrote diff_t*.jpg (white circle = YOLO-interp rider position)")
