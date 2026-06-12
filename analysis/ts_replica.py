"""Faithful replica of the TS analyze.ts blob pipeline (480x270 box
blurs, same constants) to find where it diverges from the validated
Gaussian version in v4.py."""

import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from pipeline import detect_pass, fill_gaps  # noqa: E402
from v3 import pass_a  # noqa: E402

GW, GH = 960, 540
DS = 2
BW, BH = GW // DS, GH // DS
SEARCH = 40

clip = sys.argv[1]
quality_thresh = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
sur_r = int(sys.argv[3]) if len(sys.argv) > 3 else 4

A = pass_a(clip)  # 1024x512 transforms; rescale to 960x540 analysis px
Ms = A["Ms"].copy()
Ms[:, 0, 2] *= GW / 1024
Ms[:, 1, 2] *= GH / 512
fps = float(A["fps"][0])
W, H = int(A["size"][0]), int(A["size"][1])
toSrc = W / GW

yolo = detect_pass(clip)
yts = np.array([s["t"] for s in yolo])
ycx = fill_gaps([s["box"]["cx"] if s["box"] else np.nan for s in yolo])
ycy = fill_gaps([s["box"]["cy"] if s["box"] else np.nan for s in yolo])
yscore = np.array([s["box"]["score"] if s["box"] else 0.0 for s in yolo])


def box_blur(src, r):
    k = 2 * r + 1
    return cv2.blur(src, (k, k), borderType=cv2.BORDER_REPLICATE)


cap = cv2.VideoCapture(clip)
prev = None
px, py = ycx[0] / toSrc / DS, ycy[0] / toSrc / DS
vx = vy = 0.0
miss = 0
res_hist = []
i = 0
ts_l, x_l, y_l, found_l, q_l = [], [], [], [], []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    t = i / fps
    g = cv2.cvtColor(cv2.resize(frame, (GW, GH)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    if prev is not None:
        M = Ms[min(i - 1, len(Ms) - 1)]
        # carry by camera (analysis px -> ds px)
        ax, ay = px * DS, py * DS
        px = (M[0, 0] * ax + M[0, 1] * ay + M[0, 2]) / DS
        py = (M[1, 0] * ax + M[1, 1] * ay + M[1, 2]) / DS
        vx2 = M[0, 0] * vx + M[0, 1] * vy
        vy2 = M[1, 0] * vx + M[1, 1] * vy
        vx, vy = vx2, vy2
        aligned = cv2.warpAffine(prev, M, (GW, GH))
        diff_full = cv2.absdiff(g, aligned)
        diff = cv2.resize(diff_full, (BW, BH), interpolation=cv2.INTER_AREA)
        m = 4
        diff[:m, :] = 0
        diff[-m:, :] = 0
        diff[:, :m] = 0
        diff[:, -m:] = 0
        mean = diff.mean()
        res_hist.append(mean)
        base = np.median(res_hist[-90:])
        align_bad = mean > max(2.2 * base, 25)
        small = box_blur(box_blur(diff, 1), 1)
        large = box_blur(box_blur(diff, sur_r), sur_r)
        resp = small - 0.8 * large
        cxp = min(max(px + vx, 0), BW - 1)
        cyp = min(max(py + vy, 0), BH - 1)
        s_eff = min(SEARCH * (1 + miss / 15), 150)
        yy, xx = np.mgrid[0:BH, 0:BW]
        score = resp * np.exp(-((xx - cxp) ** 2 + (yy - cyp) ** 2) / (2 * s_eff**2))
        score[:m, :] = -1e9
        score[-m:, :] = -1e9
        score[:, :m] = -1e9
        score[:, -m:] = -1e9
        k = int(np.argmax(score))
        bx, by = k % BW, k // BW
        quality = float(resp[by, bx])
        q_l.append(quality)
        found = quality > quality_thresh and not align_bad
        if found:
            r = 6
            x0, x1 = max(bx - r, 0), min(bx + r, BW)
            y0, y1 = max(by - r, 0), min(by + r, BH)
            patch = np.clip(resp[y0:y1, x0:x1], 0, None)
            tot = patch.sum()
            if tot > 0:
                gy, gx = np.mgrid[y0:y1, x0:x1]
                bx = float((patch * gx).sum() / tot)
                by = float((patch * gy).sum() / tot)
            vx = 0.6 * vx + 0.4 * (bx - px)
            vy = 0.6 * vy + 0.4 * (by - py)
            px, py = bx, by
            miss = 0
        else:
            miss += 1
            vx *= 0.9
            vy *= 0.9
            px = min(max(px + vx, 0), BW - 1)
            py = min(max(py + vy, 0), BH - 1)
        ys = np.interp(t, yts, yscore)
        if ys > 0.3:
            yx = np.interp(t, yts, ycx) / toSrc / DS
            yyc = np.interp(t, yts, ycy) / toSrc / DS
            if np.hypot(yx - px, yyc - py) < 100 / DS:
                px, py = 0.7 * px + 0.3 * yx, 0.7 * py + 0.3 * yyc
        ts_l.append(t)
        x_l.append(px * DS * toSrc)
        y_l.append(py * DS * toSrc)
        found_l.append(found)
    prev = g
    i += 1
cap.release()

found_arr = np.array(found_l)
q_arr = np.array(q_l)
print(f"TS-replica (thresh={quality_thresh}, surround r={sur_r}):")
print(f"  blob found: {found_arr.mean() * 100:.0f}%  | quality: median {np.median(q_arr):.1f}, p25 {np.percentile(q_arr, 25):.1f}, p75 {np.percentile(q_arr, 75):.1f}")
errs = []
for s in yolo:
    if s["box"] and s["box"]["score"] > 0.4:
        mx = np.interp(s["t"], ts_l, x_l)
        my = np.interp(s["t"], ts_l, y_l)
        errs.append(np.hypot(mx - s["box"]["cx"], my - s["box"]["cy"]))
print(f"  vs confident YOLO: median {np.median(errs):.0f}px p90 {np.percentile(errs, 90):.0f}px")
for lo in range(0, int(ts_l[-1]), 10):
    seg = found_arr[(np.array(ts_l) >= lo) & (np.array(ts_l) < lo + 10)]
    print(f"  t={lo:3d}-{lo + 10:3d}s found {seg.mean() * 100:3.0f}%")
