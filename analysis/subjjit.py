"""THE metric that matches what the user sees as 'shaky': the rider's
position WITHIN THE OUTPUT FRAME over time, high-passed. If the subject
bounces in the frame it reads as shaky even on a stable background.
Computed analytically (track -> crop transform -> output position) so we
can sweep configs without rendering each one."""

import sys

import numpy as np

sys.path.insert(0, ".")
import v4 as V  # noqa: E402
import v5 as V5  # noqa: E402
from pipeline import detect_pass, gaussian_smooth  # noqa: E402
from v3 import apply_a, pass_a  # noqa: E402


def subject_output_series(track, path, Dse, fps, W, H, outW=1920, outH=1080):
    """Rider (x,y) in the OUTPUT frame, per source frame."""
    ox, oy = [], []
    for j, t in enumerate(track["ts"]):
        D = Dse[min(j + 1, len(Dse) - 1)]
        sxp = np.interp(t, path["ts"], path["cx"])
        syp = np.interp(t, path["ts"], path["cy"])
        cw = np.interp(t, path["ts"], path["cropW"])
        ch_ = cw * H / W
        qx, qy = apply_a(D, sxp, syp)
        qx = min(max(qx, cw / 2), W - cw / 2)
        qy = min(max(qy, ch_ / 2), H - ch_ / 2)
        th, sc, _, _ = V.decompose(D[None])
        # rider in source -> output: translate by crop, scale, rotate
        rx, ry = track["x"][j], track["y"][j]
        dx, dy = rx - qx, ry - qy
        c, s = np.cos(-th[0]), np.sin(-th[0])
        k = outW / cw
        ox.append((dx * c - dy * s) * k + outW / 2)
        oy.append((dx * s + dy * c) * k + outH / 2)
    return np.array(ox), np.array(oy)


def hp(v, fps, win_s):
    k = max(3, int(fps * win_s) | 1)
    pad = k // 2
    sm = np.convolve(np.pad(v, pad, mode="edge"), np.ones(k) / k, mode="valid")
    return v - sm


def crop_center_output(path, Dse, fps, W, H, outW=1920, outH=1080):
    """The crop window's own motion -> background shake proxy: high-freq
    jitter of where the crop samples from the source."""
    qx_l, qy_l, th_l = [], [], []
    nfr = len(Dse)
    for j in range(nfr):
        t = j / fps
        D = Dse[j]
        sxp = np.interp(t, path["ts"], path["cx"])
        syp = np.interp(t, path["ts"], path["cy"])
        cw = np.interp(t, path["ts"], path["cropW"])
        ch_ = cw * H / W
        qx, qy = apply_a(D, sxp, syp)
        qx = min(max(qx, cw / 2), W - cw / 2)
        qy = min(max(qy, ch_ / 2), H - ch_ / 2)
        th, sc, _, _ = V.decompose(D[None])
        k = outW / cw
        qx_l.append(qx * k)
        qy_l.append(qy * k)
        th_l.append(th[0])
    return np.array(qx_l), np.array(qy_l), np.array(th_l)


def evaluate(clip, label, track, path, Dse, fps, W, H):
    ox, oy = subject_output_series(track, path, Dse, fps, W, H)
    jit = np.hypot(hp(ox, fps, 0.4), hp(oy, fps, 0.4)).std()
    offc = np.hypot(ox - 1920 / 2, oy - 1080 / 2)
    edge = (offc > 1080 * 0.42).mean() * (len(ox) / fps)
    qx, qy, th = crop_center_output(path, Dse, fps, W, H)
    bg = np.hypot(hp(qx, fps, 0.15), hp(qy, fps, 0.15)).std()
    print(
        f"  [{label:22s}] subject-bounce {jit:5.1f}px | bg-shake {bg:4.1f}px | rider-near-edge {edge:4.1f}s"
    )
    return jit, bg, edge


def build_path_tight(track, Ds, fps, W, H, leash_x, leash_y, ref_smooth, pad=V.PAD_FACTOR, widen=None):
    """v2-style iterative-leash path with tunable leash + reference
    smoothing, NO correction cap (cap was the shake regression)."""
    ts = np.arange(0, track["ts"][-1], V.SAMPLE_DT)
    ph = np.interp(ts, track["ts"], gaussian_smooth(np.asarray(track["h"], float), fps * 0.5))
    sigma = V.SMOOTH_SEC / V.SAMPLE_DT
    sph = gaussian_smooth(V.median_filter(ph), sigma * 2)
    wfac = np.interp(ts, widen[0], widen[1]) if widen is not None else 1.0
    crop_h = np.clip(sph * pad * wfac, H / V.MAX_ZOOM, H)
    crop_w = crop_h * (W / H)
    n = len(track["ts"])
    rx = np.empty(n)
    ry = np.empty(n)
    for j in range(n):
        Di = V.invert(Ds[min(j + 1, len(Ds) - 1)])
        rx[j], ry[j] = apply_a(Di, track["x"][j], track["y"][j])
    cx = gaussian_smooth(V.hampel(np.interp(ts, track["ts"], rx)), ref_smooth)
    cy = gaussian_smooth(V.hampel(np.interp(ts, track["ts"], ry)), ref_smooth)
    lim_x = crop_w * leash_x / 2
    lim_y = crop_h * leash_y / 2
    mX, mY = 0.02 * W, 0.025 * H
    lo_x, hi_x = crop_w / 2 + mX, W - crop_w / 2 - mX
    lo_y, hi_y = crop_h / 2 + mY, H - crop_h / 2 - mY

    def constrain(vx_, vy_):
        return (
            np.where(hi_x < lo_x, W / 2, np.clip(vx_, lo_x, hi_x)),
            np.where(hi_y < lo_y, H / 2, np.clip(vy_, lo_y, hi_y)),
        )

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
    return {"ts": ts, "cx": sx_, "cy": sy_, "cropW": crop_w, "cropH": crop_h}, Ds


if __name__ == "__main__":
    clip = sys.argv[1]
    A = pass_a(clip)
    fps = float(A["fps"][0])
    W, H = int(A["size"][0]), int(A["size"][1])
    Ms = V.to_source_Ms(A["Ms"], W, H)
    Ds0 = V.residuals_incremental(Ms, fps)
    yolo = detect_pass(clip)
    blob = V.track_blobs(clip, A, yolo)

    speeds = np.hypot(Ms[:, 0, 2], Ms[:, 1, 2])
    r = int(fps * 0.3)
    env = np.array([speeds[max(0, i - r) : i + r + 1].max() for i in range(len(speeds))])
    sp = gaussian_smooth(env, fps * 0.3)
    x01 = np.clip((sp - 0.021 * W) / (0.031 * W), 0, 1)
    widen = (np.arange(len(sp)) / fps, 1 + 0.8 * (x01 * x01 * (3 - 2 * x01)))

    print(f"{clip.split(chr(92))[-1]}  (blob {blob['found'].mean()*100:.0f}%)")
    p_v7, D_v7 = V.build_path_v4(blob, Ds0, fps, W, H, widen=widen)
    evaluate(clip, "current (v7 leash)", blob, p_v7, D_v7, fps, W, H)

    # LOCK-ON: crop center = rider position, gaussian-smoothed by `foll`
    # seconds (no leash drift). Sweep how tight the follow is.
    csrt = V5.track_csrt(clip, yolo, blob, fps, W, H)
    for tname, track in [("blob", blob), ("csrt", csrt)]:
        ph = gaussian_smooth(np.asarray(track["h"], float), fps * 0.5)
        crop_h_f = np.clip(ph * V.PAD_FACTOR, H / V.MAX_ZOOM, H)
        for foll in [0.15, 0.3, 0.5, 0.8]:
            sig = foll * fps
            cxf = gaussian_smooth(V.hampel(np.asarray(track["x"], float)), sig)
            cyf = gaussian_smooth(V.hampel(np.asarray(track["y"], float)), sig)
            ts = np.arange(len(track["x"])) / fps
            cw = crop_h_f * (W / H)
            cxf = np.clip(cxf, cw / 2, W - cw / 2)
            cyf = np.clip(cyf, crop_h_f / 2, H - crop_h_f / 2)
            path = {"ts": ts, "cx": cxf, "cy": cyf, "cropW": cw, "cropH": crop_h_f}
            # lock-on follows the rider directly in FRAME coords, so the
            # render must NOT re-apply D translation (would double-count);
            # keep only D rotation/scale for roll/zoom de-shake
            Dlock = Ds0.copy()
            Dlock[:, 0, 2] = 0
            Dlock[:, 1, 2] = 0
            evaluate(clip, f"{tname} lock foll={foll}", track, path, Dlock, fps, W, H)
