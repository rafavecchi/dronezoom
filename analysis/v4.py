"""v4: similarity chain + center-surround blob tracking + smoothed-camera
coordinate path (handles real rotation/zoom in the footage).

Architecture being validated for the TS port:
  D_k = C_k ∘ Ĉ_k⁻¹            per-frame shake residual transform
  r̂_k = D_k⁻¹(r_k)             rider position with shake removed
  S(t) = smooth+leash(r̂)       virtual camera path (intended coords)
  q_k = D_k(S(t_k))            crop center in the actual frame
  render: counter-rotate/scale by D_k's θ, s about q
"""

import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from pipeline import detect_pass, fill_gaps, gaussian_smooth, median_filter  # noqa: E402
from v3 import AW, AH, apply_a, chain, compose, decompose, invert, pass_a  # noqa: E402

PAD_FACTOR = 5.0
SMOOTH_SEC = 1.0
MAX_ZOOM = 4.0
LEASH_X = 0.45
LEASH_Y = 0.35
SAMPLE_DT = 0.2


def residuals_incremental(Ms_src, fps, smooth_sec=SMOOTH_SEC, leak=0.005):
    """Shake residual D_k from per-frame increments.

    Decompose each frame-to-frame transform into (dθ, dlog s, dt), smooth
    those small increment series to get the intended motion M̂, and evolve
    D_k = M_k ∘ D_{k-1} ∘ M̂_k⁻¹ with a slow leak toward identity. All
    quantities stay near zero — no global-frame blowup.
    """
    th = np.array([np.arctan2(M[1, 0], M[0, 0]) for M in Ms_src])
    ls = np.array([np.log(np.hypot(M[0, 0], M[1, 0])) for M in Ms_src])
    tx = Ms_src[:, 0, 2]
    ty = Ms_src[:, 1, 2]
    k = smooth_sec * fps
    sth, sls = gaussian_smooth(th, k), gaussian_smooth(ls, k)
    stx, sty = gaussian_smooth(tx, k), gaussian_smooth(ty, k)
    D = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    eye = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    out = [D.copy()]
    for i, M in enumerate(Ms_src):
        a, b = np.exp(sls[i]) * np.cos(sth[i]), np.exp(sls[i]) * np.sin(sth[i])
        Mhat = np.array([[a, -b, stx[i]], [b, a, sty[i]]])
        D = compose(M, compose(D, invert(Mhat)))
        D = (1 - leak) * D + leak * eye
        out.append(D.copy())
    return np.array(out)


def to_source_Ms(Ms, W, H):
    sx, sy = W / AW, H / AH
    S = np.array([[sx, 0, 0], [0, sy, 0]])
    Si = np.array([[1 / sx, 0, 0], [0, 1 / sy, 0]])
    return np.array([compose(S, compose(M, Si)) for M in Ms])


def track_blobs(clip, A, yolo_samples):
    """Center-surround response tracking on shake-aligned diffs."""
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
    SEARCH = 80
    res_hist = []
    miss = 0
    out_t, out_x, out_y, out_h, out_q = [], [], [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        g = cv2.cvtColor(cv2.resize(frame, (AW, AH)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            M = Ms[min(i - 1, len(Ms) - 1)]
            # the camera moved: carry the prediction (and velocity) with
            # it — during a whip pan the rider's frame position moves with
            # the camera, not with his own ground motion
            px, py = M[0, 0] * px + M[0, 1] * py + M[0, 2], M[1, 0] * px + M[1, 1] * py + M[1, 2]
            vx, vy = M[0, 0] * vx + M[0, 1] * vy, M[1, 0] * vx + M[1, 1] * vy
            aligned = cv2.warpAffine(prev, M, (AW, AH))
            diff = cv2.absdiff(g, aligned)
            # NOTE: luminance-normalizing the diff (to boost shadowed
            # movers) was tried and reverted — it amplified shadow noise
            # and degraded sunny segments 7px -> 113px.
            m = 8
            diff[:m, :] = 0
            diff[-m:, :] = 0
            diff[:, :m] = 0
            diff[:, -m:] = 0
            # whip pan / failed alignment: whole frame lights up — any blob
            # found in that mess is noise, so coast instead. Adaptive: the
            # clip's normal residual level varies with texture.
            cur_res = diff[m:-m, m:-m].mean()
            res_hist.append(cur_res)
            base_res = np.median(res_hist[-90:])
            align_bad = cur_res > max(2.2 * base_res, 25)
            # center-surround: compact blobs pop, elongated parallax bands cancel
            resp = cv2.GaussianBlur(diff, (0, 0), 3) - 0.8 * cv2.GaussianBlur(diff, (0, 0), 12)
            cx_pred = min(max(px + vx, 0), AW - 1)
            cy_pred = min(max(py + vy, 0), AH - 1)
            # proximity weight; search widens with the miss streak so a
            # lost track can re-acquire
            s_eff = min(SEARCH * (1 + miss / 15), 300)
            yy, xx = np.mgrid[0:AH, 0:AW]
            prox = np.exp(-((xx - cx_pred) ** 2 + (yy - cy_pred) ** 2) / (2 * s_eff**2))
            score = resp * prox
            k = int(np.argmax(score))
            bx, by = k % AW, k // AW
            quality = float(resp[by, bx])
            found = quality > 4.0 and not align_bad
            if found:
                # refine: centroid of strong response near peak
                r = 12
                x0, x1 = max(bx - r, 0), min(bx + r, AW)
                y0, y1 = max(by - r, 0), min(by + r, AH)
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
                # height: extent of diff > half peak around blob
                col = diff[max(int(by) - 20, 0) : int(by) + 20, max(int(bx) - 4, 0) : int(bx) + 4]
                hh = max(6.0, float((col > diff[int(by), int(bx)] * 0.4).sum(axis=0).max()))
                out_h.append(hh * sy)
            else:
                miss += 1
                # decay own-motion fast on miss: a rider who stopped (e.g.
                # direction switch) is stationary in world coords, and the
                # camera-carry already handles apparent motion
                vx *= 0.75
                vy *= 0.75
                px = min(max(px + vx, 0), AW)
                py = min(max(py + vy, 0), AH)
                out_h.append(out_h[-1] if out_h else 80.0)
            # YOLO re-anchor at ACTUAL confident samples only (never the
            # interpolation between them — it cuts corners through gaps
            # and carries box noise). Rescue-snap when the blob clearly
            # left the rider; otherwise the merest nudge.
            near = np.abs(yts - t) < 0.5 / fps
            if near.any():
                j = int(np.argmax(near))
                if yscore[j] > 0.45:
                    yx, yyc = ycx[j] / sx, ycy[j] / sy
                    d = np.hypot(yx - px, yyc - py)
                    if d > 60:
                        px, py = yx, yyc
                        miss = 0
                    else:
                        px, py = 0.9 * px + 0.1 * yx, 0.9 * py + 0.1 * yyc
            out_t.append(t)
            out_x.append(px * sx)
            out_y.append(py * sy)
            out_q.append(found)
        prev = g
        i += 1
    cap.release()
    return {
        "ts": np.array(out_t),
        "x": np.array(out_x),
        "y": np.array(out_y),
        "h": np.array(out_h),
        "found": np.array(out_q),
    }


def build_path_v4(track, Ds, fps, W, H, pad=PAD_FACTOR, smooth_sec=SMOOTH_SEC, widen=None):
    n = len(track["ts"])
    rx = np.empty(n)
    ry = np.empty(n)
    for j in range(n):
        Di = invert(Ds[min(j + 1, len(Ds) - 1)])
        rx[j], ry[j] = apply_a(Di, track["x"][j], track["y"][j])
    ts = np.arange(0, track["ts"][-1], SAMPLE_DT)
    cx = median_filter(np.interp(ts, track["ts"], rx))
    cy = median_filter(np.interp(ts, track["ts"], ry))
    ph = np.interp(ts, track["ts"], gaussian_smooth(track["h"], fps * 0.5))
    sigma = smooth_sec / SAMPLE_DT
    sph = gaussian_smooth(median_filter(ph), sigma * 2)
    wfac = np.interp(ts, widen[0], widen[1]) if widen is not None else 1.0
    crop_h = np.clip(sph * pad * wfac, H / MAX_ZOOM, H)
    crop_w = crop_h * (W / H)
    lim_x = crop_w * LEASH_X / 2
    lim_y = crop_h * LEASH_Y / 2
    mX, mY = 0.02 * W, 0.025 * H
    lo_x, hi_x = crop_w / 2 + mX, W - crop_w / 2 - mX
    lo_y, hi_y = crop_h / 2 + mY, H - crop_h / 2 - mY

    def constrain(vx_, vy_):
        ox = np.where(hi_x < lo_x, W / 2, np.clip(vx_, lo_x, hi_x))
        oy = np.where(hi_y < lo_y, H / 2, np.clip(vy_, lo_y, hi_y))
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
    return {"ts": ts, "cx": sx_, "cy": sy_, "cropW": crop_w, "cropH": crop_h}


def render_v4(clip, out_path, path, Ds, fps, W, H, use_rot=True):
    cap = cv2.VideoCapture(clip)
    outW, outH = 1920, 1080
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (outW, outH))
    th_d, sc_d, _, _ = decompose(Ds)
    if not use_rot:
        th_d = np.zeros_like(th_d)
        sc_d = np.ones_like(sc_d)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        k = min(i, len(Ds) - 1)
        D = Ds[k]
        sx_ = np.interp(t, path["ts"], path["cx"])
        sy_ = np.interp(t, path["ts"], path["cy"])
        cw = np.interp(t, path["ts"], path["cropW"])
        qx, qy = apply_a(D, sx_, sy_)
        qx = min(max(qx, cw / 2), W - cw / 2)
        ch_ = cw * H / W
        qy = min(max(qy, ch_ / 2), H - ch_ / 2)
        zoom = outW / (cw * sc_d[k])
        M = cv2.getRotationMatrix2D((float(qx), float(qy)), np.degrees(th_d[k]), zoom)
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
    Ms_src = to_source_Ms(A["Ms"], W, H)
    Ds = residuals_incremental(Ms_src, fps)
    th_d, sc_d, tx_d, ty_d = decompose(Ds)
    print(f"residual transform: |roll| p95 {np.degrees(np.percentile(np.abs(np.unwrap(th_d)), 95)):.2f}deg, |t| p95 {np.percentile(np.hypot(tx_d, ty_d), 95):.0f}px")

    yolo = detect_pass(clip)
    track = track_blobs(clip, A, yolo)
    print(f"blob track: {track['found'].mean() * 100:.0f}% frames")
    errs = []
    for s in yolo:
        if s["box"] and s["box"]["score"] > 0.4:
            mx = np.interp(s["t"], track["ts"], track["x"])
            my = np.interp(s["t"], track["ts"], track["y"])
            errs.append(np.hypot(mx - s["box"]["cx"], my - s["box"]["cy"]))
    print(f"vs confident YOLO ({len(errs)}): median {np.median(errs):.0f}px p90 {np.percentile(errs, 90):.0f}px")

    # zoom-out-on-pan: widen the crop when the camera moves violently —
    # the cinematographer's "go wide when the action gets fast"
    speeds = np.hypot(Ms_src[:, 0, 2], Ms_src[:, 1, 2])
    # envelope: rolling max keeps whip magnitude, then smooth the shape
    r = int(fps * 0.3)
    env = np.array([speeds[max(0, i - r) : i + r + 1].max() for i in range(len(speeds))])
    sp_s = gaussian_smooth(env, fps * 0.3)
    x01 = np.clip((sp_s - 40) / (100 - 40), 0, 1)
    wfac = 1 + 0.8 * (x01 * x01 * (3 - 2 * x01))  # smoothstep
    widen = (np.arange(len(wfac)) / fps, wfac)
    print(f"widen factor: median {np.median(wfac):.2f}, p95 {np.percentile(wfac, 95):.2f}, max {wfac.max():.2f}")

    path = build_path_v4(track, Ds, fps, W, H, widen=widen)
    render_v4(clip, "sim-v4.mp4", path, Ds, fps, W, H)
    render_v4(clip, "sim-v4-norot.mp4", path, Ds, fps, W, H, use_rot=False)
    print("rendered sim-v4.mp4 + sim-v4-norot.mp4")

    offs = []
    for j, t in enumerate(track["ts"]):
        D = Ds[min(j + 1, len(Ds) - 1)]
        sxp = np.interp(t, path["ts"], path["cx"])
        syp = np.interp(t, path["ts"], path["cy"])
        cw = np.interp(t, path["ts"], path["cropW"])
        qx, qy = apply_a(D, sxp, syp)
        qx = min(max(qx, cw / 2), W - cw / 2)
        ch_ = cw * H / W
        qy = min(max(qy, ch_ / 2), H - ch_ / 2)
        offs.append((abs(track["x"][j] - qx) / (cw / 2), abs(track["y"][j] - qy) / (ch_ / 2)))
    offs = np.array(offs)
    print(f"rider offset (frac half-crop): x med {np.median(offs[:, 0]):.2f} p95 {np.percentile(offs[:, 0], 95):.2f} | y med {np.median(offs[:, 1]):.2f} p95 {np.percentile(offs[:, 1], 95):.2f}")
