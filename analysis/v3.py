"""v3 prototype: grid phase correlation -> similarity transform chain ->
motion-blob rider track -> leashed world path -> render + metrics.

This is the candidate architecture to port back into the TS app.
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from pipeline import detect_pass, fill_gaps, gaussian_smooth, median_filter  # noqa: E402

AW, AH = 1024, 512
WIN = 256
PAD_FACTOR = 5.0
SMOOTH_SEC = 1.0
MAX_ZOOM = 4.0
LEASH_X = 0.45
LEASH_Y = 0.35
SAMPLE_DT = 0.2
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def grid_windows():
    xs = [42, 384, 726]
    ys = [0, 256]
    return [(x, y) for y in ys for x in xs]


def fit_similarity(centers, shifts, weights):
    """Fit p' = A p + t with A=[[a,-b],[b,a]] to p'=p+shift, weighted."""
    rows, rhs, ws = [], [], []
    for (px, py), (dx, dy), w in zip(centers, shifts, weights):
        rows.append([px, -py, 1, 0])
        rhs.append(px + dx)
        ws.append(w)
        rows.append([py, px, 0, 1])
        rhs.append(py + dy)
        ws.append(w)
    A = np.array(rows) * np.array(ws)[:, None]
    b = np.array(rhs) * np.array(ws)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, bb, tx, ty = sol
    return np.array([[a, -bb, tx], [bb, a, ty]])


def pass_a(clip):
    """Per-frame similarity transforms + rider motion blobs."""
    key = os.path.join(CACHE, f"v3a_{os.path.basename(clip)}.npz")
    if os.path.exists(key):
        z = np.load(key)
        return {k: z[k] for k in z.files}
    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    win = cv2.createHanningWindow((WIN, WIN), cv2.CV_32F)
    sx, sy = W / AW, H / AH
    prev = None
    Ms = []  # per-frame 2x3 (analysis px), frame k-1 -> k
    diffs_meta = []  # (t, blob features) gathered later in pass B
    grays = None
    i = 0
    # store residual diff frames? too big; do blob tracking inline in pass B
    # here we only compute transforms; cheap to redo diff in pass B with M known
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(frame, (AW, AH)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            centers, shifts, weights = [], [], []
            for x0, y0 in grid_windows():
                (dx, dy), pk = cv2.phaseCorrelate(
                    prev[y0 : y0 + WIN, x0 : x0 + WIN], g[y0 : y0 + WIN, x0 : x0 + WIN], win
                )
                if pk > 0.03 and abs(dx) < WIN * 0.3 and abs(dy) < WIN * 0.3:
                    centers.append((x0 + WIN / 2, y0 + WIN / 2))
                    shifts.append((dx, dy))
                    weights.append(pk)
            if len(centers) >= 3:
                M = fit_similarity(centers, shifts, weights)
                # outlier rejection: drop windows with residual > 3px, refit
                keep_c, keep_s, keep_w = [], [], []
                for c, s, w in zip(centers, shifts, weights):
                    pred = M @ np.array([c[0], c[1], 1.0])
                    res = np.hypot(pred[0] - (c[0] + s[0]), pred[1] - (c[1] + s[1]))
                    if res < 3:
                        keep_c.append(c)
                        keep_s.append(s)
                        keep_w.append(w)
                if len(keep_c) >= 3 and len(keep_c) < len(centers):
                    M = fit_similarity(keep_c, keep_s, keep_w)
            else:
                M = np.array([[1.0, 0, 0], [0, 1.0, 0]])
            # Sanity clamp: a frame pair can't rotate >2deg or zoom >3%.
            # Garbage fits (whip pans, low texture) otherwise poison the
            # cumulative chain forever. Fall back to median translation.
            a_, b_ = M[0, 0], M[1, 0]
            th_ = np.arctan2(b_, a_)
            sc_ = np.hypot(a_, b_)
            if abs(th_) > 0.035 or abs(sc_ - 1) > 0.03 or np.hypot(M[0, 2], M[1, 2]) > 200:
                if shifts:
                    mdx = float(np.median([s[0] for s in shifts]))
                    mdy = float(np.median([s[1] for s in shifts]))
                else:
                    mdx = mdy = 0.0
                if abs(mdx) > WIN * 0.3 or abs(mdy) > WIN * 0.3:
                    mdx = mdy = 0.0
                M = np.array([[1.0, 0, mdx], [0, 1.0, mdy]])
            Ms.append(M)
        prev = g
        i += 1
    cap.release()
    Ms = np.array(Ms)  # [n-1, 2, 3]
    out = {"Ms": Ms, "fps": np.array([fps]), "size": np.array([W, H])}
    np.savez(key, **out)
    return out


def compose(M, C):
    """Return affine M∘C (apply C then M), both 2x3."""
    R = np.zeros((2, 3))
    R[:, :2] = M[:, :2] @ C[:, :2]
    R[:, 2] = M[:, :2] @ C[:, 2] + M[:, 2]
    return R


def invert(C):
    Ai = np.linalg.inv(C[:, :2])
    R = np.zeros((2, 3))
    R[:, :2] = Ai
    R[:, 2] = -Ai @ C[:, 2]
    return R


def apply_a(C, x, y):
    return C[0, 0] * x + C[0, 1] * y + C[0, 2], C[1, 0] * x + C[1, 1] * y + C[1, 2]


def pass_b(clip, A, yolo_samples):
    """Blob track using aligned diffs; YOLO re-anchors when confident."""
    Ms = A["Ms"]
    fps = float(A["fps"][0])
    W, H = int(A["size"][0]), int(A["size"][1])
    sx, sy = W / AW, H / AH
    yts = np.array([s["t"] for s in yolo_samples])
    ycx = fill_gaps([s["box"]["cx"] if s["box"] else np.nan for s in yolo_samples])
    ycy = fill_gaps([s["box"]["cy"] if s["box"] else np.nan for s in yolo_samples])
    yscore = np.array([s["box"]["score"] if s["box"] else 0.0 for s in yolo_samples])

    cap = cv2.VideoCapture(clip)
    prev = None
    px, py = ycx[0] / sx, ycy[0] / sy
    vx = vy = 0.0
    SEARCH = 90
    t_arr, x_arr, y_arr, h_arr, ok_arr = [], [], [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        g = cv2.cvtColor(cv2.resize(frame, (AW, AH)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            M = Ms[min(i - 1, len(Ms) - 1)]
            aligned = cv2.warpAffine(prev, M, (AW, AH))
            diff = cv2.absdiff(g, aligned)
            diff = cv2.GaussianBlur(diff, (5, 5), 0)
            m = 6
            diff[:m, :] = 0
            diff[-m:, :] = 0
            diff[:, :m] = 0
            diff[:, -m:] = 0
            cx_pred = min(max(px + vx, SEARCH), AW - SEARCH)
            cy_pred = min(max(py + vy, SEARCH), AH - SEARCH)
            x0, x1 = int(cx_pred - SEARCH), int(cx_pred + SEARCH)
            y0, y1 = int(cy_pred - SEARCH), int(cy_pred + SEARCH)
            winr = diff[y0:y1, x0:x1]
            noise = np.median(diff[diff > 0.5]) if (diff > 0.5).any() else 1.0
            mask = (winr > max(6.0, 3.5 * noise)).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
            found = False
            if n > 1:
                # blob nearest prediction with decent area
                best_k, best_cost = -1, 1e18
                for k in range(1, n):
                    if stats[k, cv2.CC_STAT_AREA] < 10:
                        continue
                    bx, by = cents[k]
                    cost = np.hypot(x0 + bx - cx_pred, y0 + by - cy_pred) - 2 * np.sqrt(
                        stats[k, cv2.CC_STAT_AREA]
                    )
                    if cost < best_cost:
                        best_cost, best_k = cost, k
                if best_k > 0:
                    bx, by = cents[best_k]
                    nx, ny = x0 + bx, y0 + by
                    vx = 0.6 * vx + 0.4 * (nx - px)
                    vy = 0.6 * vy + 0.4 * (ny - py)
                    px, py = nx, ny
                    h_arr.append(stats[best_k, cv2.CC_STAT_HEIGHT] * sy)
                    found = True
            if not found:
                px += vx
                py += vy
                px = min(max(px, 0), AW)
                py = min(max(py, 0), AH)
                h_arr.append(h_arr[-1] if h_arr else 80.0)
            # YOLO re-anchor when confident
            ys = np.interp(t, yts, yscore)
            if ys > 0.4:
                yx, yy = np.interp(t, yts, ycx) / sx, np.interp(t, yts, ycy) / sy
                if np.hypot(yx - px, yy - py) < 120:
                    px, py = 0.7 * px + 0.3 * yx, 0.7 * py + 0.3 * yy
            t_arr.append(t)
            x_arr.append(px * sx)
            y_arr.append(py * sy)
            ok_arr.append(found)
        prev = g
        i += 1
    cap.release()
    return {
        "ts": np.array(t_arr),
        "x": np.array(x_arr),
        "y": np.array(y_arr),
        "h": np.array(h_arr),
        "found": np.array(ok_arr),
    }


def chain(Ms, W, H):
    """Cumulative transforms in SOURCE px: C[k] maps world(frame0)->frame k."""
    sx, sy = W / AW, H / AH
    S = np.array([[sx, 0, 0], [0, sy, 0]])
    Si = np.array([[1 / sx, 0, 0], [0, 1 / sy, 0]])
    C = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    out = [C.copy()]
    for M in Ms:
        Msrc = compose(S, compose(M, Si))  # analysis-px transform in source px
        C = compose(Msrc, C)
        out.append(C.copy())
    return np.array(out)


def decompose(Cs):
    a, b = Cs[:, 0, 0], Cs[:, 1, 0]
    theta = np.arctan2(b, a)
    scale = np.hypot(a, b)
    return theta, scale, Cs[:, 0, 2], Cs[:, 1, 2]


def build_world_path(track, Cs, fps, W, H, pad=PAD_FACTOR, smooth_sec=SMOOTH_SEC):
    # world coords of rider track (per frame)
    n = len(track["ts"])
    wx = np.empty(n)
    wy = np.empty(n)
    for j in range(n):
        Ci = invert(Cs[min(j + 1, len(Cs) - 1)])
        wx[j], wy[j] = apply_a(Ci, track["x"][j], track["y"][j])
    # sample to grid
    ts = np.arange(0, track["ts"][-1], SAMPLE_DT)
    cx = np.interp(ts, track["ts"], wx)
    cy = np.interp(ts, track["ts"], wy)
    ph = np.interp(ts, track["ts"], gaussian_smooth(track["h"], fps))
    sigma = smooth_sec / SAMPLE_DT
    cx = median_filter(cx)
    cy = median_filter(cy)
    sph = gaussian_smooth(median_filter(ph), sigma * 2)
    crop_h = np.clip(sph * pad, H / MAX_ZOOM, H)
    crop_w = crop_h * (W / H)
    lim_x = crop_w * LEASH_X / 2
    lim_y = crop_h * LEASH_Y / 2
    # smoothed camera for constrain
    theta, scale, tx, ty = decompose(Cs)
    k_s = smooth_sec * fps
    sm = {
        "theta": gaussian_smooth(theta, k_s),
        "scale": gaussian_smooth(scale, k_s),
        "tx": gaussian_smooth(tx, k_s),
        "ty": gaussian_smooth(ty, k_s),
    }
    cam_ts = np.arange(len(Cs)) / fps

    def smooth_C_at(t):
        th = np.interp(t, cam_ts, sm["theta"])
        sc = np.interp(t, cam_ts, sm["scale"])
        txx = np.interp(t, cam_ts, sm["tx"])
        tyy = np.interp(t, cam_ts, sm["ty"])
        a, b = sc * np.cos(th), sc * np.sin(th)
        return np.array([[a, -b, txx], [b, a, tyy]])

    mX, mY = 0.02 * W, 0.025 * H

    def constrain(sx_, sy_):
        ox = sx_.copy()
        oy = sy_.copy()
        for j, t in enumerate(ts):
            Cm = smooth_C_at(t)
            fx_, fy_ = apply_a(Cm, sx_[j], sy_[j])
            lo_x, hi_x = crop_w[j] / 2 + mX, W - crop_w[j] / 2 - mX
            lo_y, hi_y = crop_h[j] / 2 + mY, H - crop_h[j] / 2 - mY
            cfx = W / 2 if hi_x < lo_x else min(max(fx_, lo_x), hi_x)
            cfy = H / 2 if hi_y < lo_y else min(max(fy_, lo_y), hi_y)
            if cfx != fx_ or cfy != fy_:
                ox[j], oy[j] = apply_a(invert(Cm), cfx, cfy)
        return ox, oy

    sx_ = gaussian_smooth(cx, sigma)
    sy_ = gaussian_smooth(cy, sigma)
    for it in range(6):
        s = sigma * 0.55 ** (it + 1)
        sx_ = np.clip(sx_, cx - lim_x, cx + lim_x)
        sy_ = np.clip(sy_, cy - lim_y, cy + lim_y)
        sx_, sy_ = constrain(sx_, sy_)
        sx_ = gaussian_smooth(sx_, s)
        sy_ = gaussian_smooth(sy_, s)
    sx_ = np.clip(sx_, cx - lim_x, cx + lim_x)
    sy_ = np.clip(sy_, cy - lim_y, cy + lim_y)
    sx_, sy_ = constrain(sx_, sy_)
    return {"ts": ts, "cx": sx_, "cy": sy_, "cropW": crop_w, "cropH": crop_h, "smooth": sm, "cam_ts": cam_ts}


def render_v3(clip, out_path, path, Cs, fps, W, H):
    cap = cv2.VideoCapture(clip)
    outW, outH = 1920, 1080
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (outW, outH))
    theta, scale, _, _ = decompose(Cs)
    sm = path["smooth"]
    cam_ts = path["cam_ts"]
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        k = min(i, len(Cs) - 1)
        C = Cs[k]
        wx = np.interp(t, path["ts"], path["cx"])
        wy = np.interp(t, path["ts"], path["cy"])
        cw = np.interp(t, path["ts"], path["cropW"])
        qx, qy = apply_a(C, wx, wy)
        qx = min(max(qx, cw / 2), W - cw / 2)
        ch_ = cw * H / W
        qy = min(max(qy, ch_ / 2), H - ch_ / 2)
        th_res = theta[k] - np.interp(t, cam_ts, sm["theta"])
        sc_res = scale[k] / max(np.interp(t, cam_ts, sm["scale"]), 1e-6)
        zoom = outW / (cw * sc_res)
        M = cv2.getRotationMatrix2D((float(qx), float(qy)), np.degrees(th_res), zoom)
        M[0, 2] += outW / 2 - qx
        M[1, 2] += outH / 2 - qy
        vw.write(cv2.warpAffine(frame, M, (outW, outH), flags=cv2.INTER_LINEAR))
        i += 1
    cap.release()
    vw.release()


if __name__ == "__main__":
    clip = sys.argv[1]
    A = pass_a(clip)
    W, H = int(A["size"][0]), int(A["size"][1])
    fps = float(A["fps"][0])
    theta, scale, tx, ty = decompose(chain(A["Ms"], W, H))
    print(f"transform chain: roll range {np.degrees(theta.min()):.1f}..{np.degrees(theta.max()):.1f} deg, scale {scale.min():.3f}..{scale.max():.3f}")

    yolo = detect_pass(clip)
    track = pass_b(clip, A, yolo)
    print(f"blob track: {track['found'].mean() * 100:.0f}% frames found")
    # compare with confident YOLO
    errs = []
    for s in yolo:
        if s["box"] and s["box"]["score"] > 0.4:
            mx = np.interp(s["t"], track["ts"], track["x"])
            my = np.interp(s["t"], track["ts"], track["y"])
            errs.append(np.hypot(mx - s["box"]["cx"], my - s["box"]["cy"]))
    if errs:
        print(f"vs confident YOLO ({len(errs)}): median {np.median(errs):.0f}px p90 {np.percentile(errs, 90):.0f}px")

    Cs = chain(A["Ms"], W, H)
    path = build_world_path(track, Cs, fps, W, H)
    render_v3(clip, "sim-v3.mp4", path, Cs, fps, W, H)
    print("rendered sim-v3.mp4")

    # centering metric
    offs = []
    for j, t in enumerate(track["ts"]):
        C = Cs[min(j + 1, len(Cs) - 1)]
        wx = np.interp(t, path["ts"], path["cx"])
        wy = np.interp(t, path["ts"], path["cy"])
        cw = np.interp(t, path["ts"], path["cropW"])
        qx, qy = apply_a(C, wx, wy)
        qx = min(max(qx, cw / 2), W - cw / 2)
        ch_ = cw * H / W
        qy = min(max(qy, ch_ / 2), H - ch_ / 2)
        offs.append((abs(track["x"][j] - qx) / (cw / 2), abs(track["y"][j] - qy) / (ch_ / 2)))
    offs = np.array(offs)
    print(f"rider offset (frac of half-crop): x med {np.median(offs[:, 0]):.2f} p95 {np.percentile(offs[:, 0], 95):.2f} | y med {np.median(offs[:, 1]):.2f} p95 {np.percentile(offs[:, 1], 95):.2f}")
