"""Annotate source frames with the tracked rider (yellow) and crop box
(green) at given times, so failures can be seen directly."""

import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from pipeline import detect_pass, gaussian_smooth  # noqa: E402
from v3 import apply_a, pass_a  # noqa: E402
from v4 import build_path_v4, decompose, residuals_incremental, to_source_Ms, track_blobs  # noqa: E402

clip = sys.argv[1]
times = [float(x) for x in sys.argv[2:]]
A = pass_a(clip)
fps = float(A["fps"][0])
W, H = int(A["size"][0]), int(A["size"][1])
Ms = to_source_Ms(A["Ms"], W, H)
Ds = residuals_incremental(Ms, fps)
yolo = detect_pass(clip)
track = track_blobs(clip, A, yolo)
speeds = np.hypot(Ms[:, 0, 2], Ms[:, 1, 2])
r = int(fps * 0.3)
env = np.array([speeds[max(0, i - r) : i + r + 1].max() for i in range(len(speeds))])
sp = gaussian_smooth(env, fps * 0.3)
x01 = np.clip((sp - 0.021 * W) / (0.031 * W), 0, 1)
widen = (np.arange(len(sp)) / fps, 1 + 0.8 * (x01 * x01 * (3 - 2 * x01)))
path, Dse = build_path_v4(track, Ds, fps, W, H, widen=widen)
yts = np.array([s["t"] for s in yolo])

cap = cv2.VideoCapture(clip)
for t in times:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, frame = cap.read()
    if not ok:
        continue
    j = min(int(t * fps), len(track["ts"]) - 1)
    # tracked rider (yellow)
    tx, ty = int(track["x"][j]), int(track["y"][j])
    cv2.circle(frame, (tx, ty), 45, (0, 255, 255), 5)
    cv2.putText(frame, "TRACK", (tx + 50, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    # crop box (green)
    D = Dse[min(j + 1, len(Dse) - 1)]
    sxp = np.interp(t, path["ts"], path["cx"])
    syp = np.interp(t, path["ts"], path["cy"])
    cw = np.interp(t, path["ts"], path["cropW"])
    ch_ = cw * H / W
    qx, qy = apply_a(D, sxp, syp)
    qx = min(max(qx, cw / 2), W - cw / 2)
    qy = min(max(qy, ch_ / 2), H - ch_ / 2)
    cv2.rectangle(
        frame,
        (int(qx - cw / 2), int(qy - ch_ / 2)),
        (int(qx + cw / 2), int(qy + ch_ / 2)),
        (0, 230, 0),
        5,
    )
    # any confident YOLO at this time (red)
    near = np.abs(yts - t) < 0.3
    for s in yolo:
        if abs(s["t"] - t) < 0.3 and s["box"] and s["box"]["score"] > 0.4:
            b = s["box"]
            cv2.rectangle(
                frame,
                (int(b["cx"] - b["w"] / 2), int(b["cy"] - b["h"] / 2)),
                (int(b["cx"] + b["w"] / 2), int(b["cy"] + b["h"] / 2)),
                (0, 0, 255),
                3,
            )
    cv2.putText(frame, f"t={t:.0f}s", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
    out = f"annot_t{t:.0f}.jpg"
    cv2.imwrite(out, cv2.resize(frame, (1280, 720)), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote", out)
cap.release()
