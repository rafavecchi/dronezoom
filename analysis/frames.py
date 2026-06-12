"""Dump consecutive frames around a timestamp from a video."""

import sys

import cv2

path, t0, count, out = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
cap = cv2.VideoCapture(path)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(t0 * fps))
for i in range(count):
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.resize(frame, (960, 540))
    cv2.imwrite(f"{out}_{i:02d}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
cap.release()
print("done", fps)
