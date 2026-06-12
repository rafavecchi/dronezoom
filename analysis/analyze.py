"""Quantify shake in the exported clip vs the source.

For each video: per-frame global translation (phase correlation on a
center crop) -> high-pass it (remove intentional camera path, keep
jitter) -> report RMS jitter per axis, plus worst seconds.
"""

import sys

import cv2
import numpy as np


def motion_series(path, max_frames=100000, downw=640):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    scale = downw / w
    dw, dh = int(w * scale), int(h * scale)
    prev = None
    dxs, dys = [], []
    win = cv2.createHanningWindow((dw, dh), cv2.CV_32F)
    i = 0
    while i < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(frame, (dw, dh)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            (dx, dy), _ = cv2.phaseCorrelate(prev, g, win)
            dxs.append(dx / scale)  # source px
            dys.append(dy / scale)
        prev = g
        i += 1
    cap.release()
    return np.array(dxs), np.array(dys), fps, (int(w), int(h)), int(n)


def jitter_report(name, dxs, dys, fps):
    # cumulative path, then high-pass: jitter = path - smooth(path)
    cx, cy = np.cumsum(dxs), np.cumsum(dys)
    k = max(3, int(round(fps * 0.5)) | 1)  # 0.5s smoothing window
    kernel = np.ones(k) / k
    pad = k // 2
    smooth = lambda v: np.convolve(np.pad(v, pad, mode="edge"), kernel, mode="valid")
    jx, jy = cx - smooth(cx), cy - smooth(cy)
    rms_x, rms_y = np.sqrt(np.mean(jx**2)), np.sqrt(np.mean(jy**2))
    p95_x, p95_y = np.percentile(np.abs(jx), 95), np.percentile(np.abs(jy), 95)
    print(f"\n== {name} ==")
    print(f"frames analyzed: {len(dxs) + 1}, fps {fps:.2f}")
    print(f"jitter RMS  px: x={rms_x:7.2f}  y={rms_y:7.2f}")
    print(f"jitter p95  px: x={p95_x:7.2f}  y={p95_y:7.2f}")
    # worst 1s windows
    sec = int(round(fps))
    mag = np.sqrt(jx**2 + jy**2)
    if len(mag) > sec:
        sums = np.convolve(mag, np.ones(sec) / sec, mode="valid")
        worst = np.argsort(sums)[-3:][::-1]
        print("worst seconds:", ", ".join(f"t={i / fps:.1f}s (avg {sums[i]:.1f}px)" for i in worst))
    return jx, jy


if __name__ == "__main__":
    src = sys.argv[1]
    exp = sys.argv[2]
    sdx, sdy, sfps, ssize, sn = motion_series(src)
    print(f"source: {ssize[0]}x{ssize[1]} {sfps:.2f}fps {sn} frames")
    jitter_report("SOURCE", sdx, sdy, sfps)
    edx, edy, efps, esize, en = motion_series(exp)
    print(f"\nexport: {esize[0]}x{esize[1]} {efps:.2f}fps {en} frames")
    jitter_report("EXPORT", edx, edy, efps)
