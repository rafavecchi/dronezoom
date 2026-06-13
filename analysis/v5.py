"""v5: CSRT+YOLO fused tracking and L1-optimal camera path.

Replaces the hand-rolled blob tracker with OpenCV's CSRT (a real visual
tracker), rescued/re-seeded by confident YOLO detections, with the
motion-blob as last-resort fallback. Replaces the iterative leash with
an L1 trend-filtered path (the professional-stabilizer formulation):
minimize w1|p'| + w2|p''| + w3|p'''| subject to |p - rider| <= leash
and frame bounds. Produces piecewise-smooth, intentional camera moves.
"""

import sys

import cv2
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

sys.path.insert(0, ".")
from pipeline import detect_pass, gaussian_smooth  # noqa: E402
from v3 import apply_a, pass_a  # noqa: E402
from v4 import (  # noqa: E402
    LEASH_X,
    LEASH_Y,
    MAX_ZOOM,
    PAD_FACTOR,
    SAMPLE_DT,
    SMOOTH_SEC,
    attenuate_residuals,
    decompose,
    hampel,
    invert,
    median_filter,
    render_v4,
    residuals_incremental,
    to_source_Ms,
    track_blobs,
)


def track_csrt(clip, yolo_samples, blob_track, fps, W, H):
    """CSRT primary, YOLO re-seed on confident disagreement, blob track
    fallback when CSRT loses lock and YOLO is silent."""
    yts = np.array([s["t"] for s in yolo_samples])
    conf = [(s["t"], s["box"]) for s in yolo_samples if s["box"] and s["box"]["score"] > 0.45]
    if not conf:
        return blob_track  # nothing to seed from; blob is all we have
    seed_t, seed_box = conf[0]

    cap = cv2.VideoCapture(clip)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    xs = np.full(n, np.nan)
    ys = np.full(n, np.nan)
    hs = np.full(n, np.nan)
    ok_arr = np.zeros(n, bool)

    tracker = None
    conf_i = 0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        # (re)seed at confident YOLO when no tracker or strong disagreement
        while conf_i < len(conf) - 1 and conf[conf_i + 1][0] <= t:
            conf_i += 1
        ct, cb = conf[conf_i]
        if tracker is None and t >= seed_t - 0.5 / fps:
            b = seed_box
            roi = (int(b["cx"] - b["w"]), int(b["cy"] - b["h"] * 0.75), int(b["w"] * 2), int(b["h"] * 1.5))
            roi = (max(roi[0], 0), max(roi[1], 0), min(roi[2], W - 1), min(roi[3], H - 1))
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, roi)
            xs[i], ys[i], hs[i] = b["cx"], b["cy"], b["h"]
            ok_arr[i] = True
            i += 1
            continue
        if tracker is not None:
            got, box = tracker.update(frame)
            if got:
                x, y, w_, h_ = box
                cx, cy = x + w_ / 2, y + h_ / 2
                xs[i], ys[i], hs[i] = cx, cy, h_ * 0.66
                ok_arr[i] = True
                # confident YOLO near in time and far in space -> re-seed
                if abs(ct - t) < 0.5 / fps and np.hypot(cb["cx"] - cx, cb["cy"] - cy) > 120:
                    roi = (
                        int(cb["cx"] - cb["w"]),
                        int(cb["cy"] - cb["h"] * 0.75),
                        int(cb["w"] * 2),
                        int(cb["h"] * 1.5),
                    )
                    roi = (max(roi[0], 0), max(roi[1], 0), min(roi[2], W - 1), min(roi[3], H - 1))
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, roi)
                    xs[i], ys[i], hs[i] = cb["cx"], cb["cy"], cb["h"]
            else:
                # lost: try re-seed from a confident YOLO at this moment
                if abs(ct - t) < 0.5 / fps:
                    roi = (
                        int(cb["cx"] - cb["w"]),
                        int(cb["cy"] - cb["h"] * 0.75),
                        int(cb["w"] * 2),
                        int(cb["h"] * 1.5),
                    )
                    roi = (max(roi[0], 0), max(roi[1], 0), min(roi[2], W - 1), min(roi[3], H - 1))
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, roi)
                    xs[i], ys[i], hs[i] = cb["cx"], cb["cy"], cb["h"]
                    ok_arr[i] = True
        i += 1
    cap.release()

    # fill gaps from the blob track (its coords are per-frame too)
    bt, bx, by, bh = blob_track["ts"], blob_track["x"], blob_track["y"], blob_track["h"]
    for j in range(n):
        if not ok_arr[j]:
            t = j / fps
            xs[j] = np.interp(t, bt, bx)
            ys[j] = np.interp(t, bt, by)
            hs[j] = np.interp(t, bt, bh)
    ts = np.arange(n) / fps
    return {"ts": ts, "x": xs, "y": ys, "h": hs, "found": ok_arr}


def l1_path(r, lim, lo, hi, dt):
    """L1 trend filtering: min w1|p'|+w2|p''|+w3|p'''| s.t. |p-r|<=lim,
    lo<=p<=hi. Variables: p (n) + slack e1,e2,e3."""
    n = len(r)
    w1, w2, w3 = 10.0, 1.0, 100.0
    n1, n2, n3 = n - 1, n - 2, n - 3
    nv = n + n1 + n2 + n3
    c = np.zeros(nv)
    c[n : n + n1] = w1
    c[n + n1 : n + n1 + n2] = w2
    c[n + n1 + n2 :] = w3
    rows = 2 * (n1 + n2 + n3)
    A = lil_matrix((rows, nv))
    b = np.zeros(rows)
    ri = 0
    for k in range(n1):  # |p_{k+1}-p_k| <= e1_k
        for sgn in (1, -1):
            A[ri, k + 1] = sgn
            A[ri, k] = -sgn
            A[ri, n + k] = -1
            ri += 1
    for k in range(n2):
        for sgn in (1, -1):
            A[ri, k + 2] = sgn
            A[ri, k + 1] = -2 * sgn
            A[ri, k] = sgn
            A[ri, n + n1 + k] = -1
            ri += 1
    for k in range(n3):
        for sgn in (1, -1):
            A[ri, k + 3] = sgn
            A[ri, k + 2] = -3 * sgn
            A[ri, k + 1] = 3 * sgn
            A[ri, k] = -sgn
            A[ri, n + n1 + n2 + k] = -1
            ri += 1
    bounds = []
    for k in range(n):
        bounds.append((max(lo[k], r[k] - lim[k]), min(hi[k], r[k] + lim[k])))
    for k in range(n, nv):
        bounds.append((0, None))
    # infeasible window guard: when the leash window conflicts with the
    # frame bounds, use the in-frame point closest to the rider (never
    # an arbitrary fallback like frame center)
    for k in range(n):
        a, bb = bounds[k]
        if a > bb:
            flo, fhi = (lo[k], hi[k]) if lo[k] <= hi[k] else ((lo[k] + hi[k]) / 2,) * 2
            p = min(max(r[k], flo), fhi)
            bounds[k] = (p, p)
    res = linprog(c, A_ub=A.tocsr(), b_ub=b, bounds=bounds, method="highs")
    if not res.success:
        return None
    return res.x[:n]


def build_path_l1(track, Ds, fps, W, H, pad=PAD_FACTOR, smooth_sec=SMOOTH_SEC, widen=None):
    ts = np.arange(0, track["ts"][-1], SAMPLE_DT)
    ph = np.interp(ts, track["ts"], gaussian_smooth(np.asarray(track["h"], float), fps * 0.5))
    sigma = smooth_sec / SAMPLE_DT
    sph = gaussian_smooth(median_filter(ph), sigma * 2)
    wfac = np.interp(ts, widen[0], widen[1]) if widen is not None else 1.0
    crop_h = np.clip(sph * pad * wfac, H / MAX_ZOOM, H)
    crop_w = crop_h * (W / H)

    frame_ts = np.arange(len(Ds)) / fps
    crop_h_f = np.interp(frame_ts, ts, crop_h)
    cap = 0.8 * np.minimum(LEASH_X * crop_h_f * (W / H), LEASH_Y * crop_h_f) / 2
    Ds = attenuate_residuals(Ds, cap, fps)

    n = len(track["ts"])
    rx = np.empty(n)
    ry = np.empty(n)
    for j in range(n):
        Di = invert(Ds[min(j + 1, len(Ds) - 1)])
        rx[j], ry[j] = apply_a(Di, track["x"][j], track["y"][j])
    cx = gaussian_smooth(hampel(np.interp(ts, track["ts"], rx)), 1.2)
    cy = gaussian_smooth(hampel(np.interp(ts, track["ts"], ry)), 1.2)

    mX, mY = 0.02 * W, 0.025 * H
    px = l1_path(cx, crop_w * LEASH_X / 2, crop_w / 2 + mX, W - crop_w / 2 - mX, SAMPLE_DT)
    py = l1_path(cy, crop_h * LEASH_Y / 2, crop_h / 2 + mY, H - crop_h / 2 - mY, SAMPLE_DT)
    if px is None or py is None:
        raise RuntimeError("L1 path solve failed")
    return {"ts": ts, "cx": px, "cy": py, "cropW": crop_w, "cropH": crop_h}, Ds


if __name__ == "__main__":
    clip = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "sim-v5.mp4"
    A = pass_a(clip)
    fps = float(A["fps"][0])
    W, H = int(A["size"][0]), int(A["size"][1])
    Ms_src = to_source_Ms(A["Ms"], W, H)
    Ds = residuals_incremental(Ms_src, fps)
    yolo = detect_pass(clip)
    blob = track_blobs(clip, A, yolo)
    track = track_csrt(clip, yolo, blob, fps, W, H)
    print(f"CSRT coverage: {track['found'].mean() * 100:.0f}% (blob fills the rest)")
    errs = [
        np.hypot(np.interp(s["t"], track["ts"], track["x"]) - s["box"]["cx"],
                 np.interp(s["t"], track["ts"], track["y"]) - s["box"]["cy"])
        for s in yolo
        if s["box"] and s["box"]["score"] > 0.35
    ]
    print(f"track vs YOLO: med {np.median(errs):.0f} p90 {np.percentile(errs, 90):.0f}px")

    speeds = np.hypot(Ms_src[:, 0, 2], Ms_src[:, 1, 2])
    r = int(fps * 0.3)
    env = np.array([speeds[max(0, i - r) : i + r + 1].max() for i in range(len(speeds))])
    sp = gaussian_smooth(env, fps * 0.3)
    x01 = np.clip((sp - 0.021 * W) / (0.031 * W), 0, 1)
    widen = (np.arange(len(sp)) / fps, 1 + 0.8 * (x01 * x01 * (3 - 2 * x01)))

    path, Dse = build_path_l1(track, Ds, fps, W, H, widen=widen)
    render_v4(clip, out, path, Dse, fps, W, H)
    print("rendered", out)

    offs = []
    for j in range(0, len(track["ts"]), 3):
        t = track["ts"][j]
        D = Dse[min(j + 1, len(Dse) - 1)]
        sxp = np.interp(t, path["ts"], path["cx"])
        syp = np.interp(t, path["ts"], path["cy"])
        cw = np.interp(t, path["ts"], path["cropW"])
        qx, qy = apply_a(D, sxp, syp)
        qx = min(max(qx, cw / 2), W - cw / 2)
        ch_ = cw * H / W
        qy = min(max(qy, ch_ / 2), H - ch_ / 2)
        offs.append(np.hypot((track["x"][j] - qx) / (cw / 2), (track["y"][j] - qy) / (ch_ / 2)))
    offs = np.array(offs)
    bad = (offs > 0.85).mean() * track["ts"][-1]
    print(f"offs p90 {np.percentile(offs, 90):.2f} p99 {np.percentile(offs, 99):.2f} | outside-crop {bad:.1f}s")
