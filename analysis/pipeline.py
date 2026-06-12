"""Python replica of the DroneZoom pipeline for offline tuning.

Mirrors detect.ts / tracker gating / path.ts / stabilize.ts closely
enough to evaluate framing + stabilization quality against a real clip,
with stage caching so iteration is fast.

Usage: python pipeline.py <clip.mp4> [--render out.mp4]
"""

import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "public", "models", "yolo11n-detect.onnx")
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)

INPUT = 640
SAMPLE_INTERVAL = 0.2
PAD_FACTOR = 5.0
SMOOTH_SEC = 1.0
MAX_ZOOM = 4.0
LEASH_X = 0.45
LEASH_Y = 0.35


# ---------- detection (ports detect.ts) ----------

_sess = None


def sess():
    global _sess
    if _sess is None:
        _sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    return _sess


def detect_persons(frame_bgr, region=None, thresh=0.35):
    h, w = frame_bgr.shape[:2]
    rx, ry, rw, rh = region if region else (0, 0, w, h)
    crop = frame_bgr[int(ry) : int(ry + rh), int(rx) : int(rx + rw)]
    ch, cw = crop.shape[:2]
    scale = min(INPUT / cw, INPUT / ch)
    dw, dh = round(cw * scale), round(ch * scale)
    dx, dy = (INPUT - dw) // 2, (INPUT - dh) // 2
    canvas = np.full((INPUT, INPUT, 3), 114, np.uint8)
    canvas[dy : dy + dh, dx : dx + dw] = cv2.resize(crop, (dw, dh))
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    inp = rgb.transpose(2, 0, 1)[None]
    out = sess().run(None, {"images": inp})[0][0]  # [84, 8400]
    scores = out[4]  # person class
    keep = scores > thresh
    if not keep.any():
        return []
    boxes = out[:4, keep].T  # cx cy w h in 640 space
    sc = scores[keep]
    dets = []
    for (bcx, bcy, bw, bh), s in zip(boxes, sc):
        dets.append(
            {
                "cx": rx + (bcx - dx) / scale,
                "cy": ry + (bcy - dy) / scale,
                "w": bw / scale,
                "h": bh / scale,
                "score": float(s),
            }
        )
    dets.sort(key=lambda d: -d["score"])
    # NMS
    out2 = []
    for d in dets:
        if all(iou(d, k) < 0.45 for k in out2):
            out2.append(d)
    return out2


def iou(a, b):
    ax1, ay1, ax2, ay2 = a["cx"] - a["w"] / 2, a["cy"] - a["h"] / 2, a["cx"] + a["w"] / 2, a["cy"] + a["h"] / 2
    bx1, by1, bx2, by2 = b["cx"] - b["w"] / 2, b["cy"] - b["h"] / 2, b["cx"] + b["w"] / 2, b["cy"] + b["h"] / 2
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    u = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / u if u > 0 else 0


def detect_pass(path, roi_first=True):
    """ROI-first tracking detection, mirroring main.ts detectAt."""
    key = os.path.join(CACHE, f"dets_roi{int(roi_first)}_{os.path.basename(path)}.npz")
    if os.path.exists(key):
        z = np.load(key, allow_pickle=True)
        return list(z["samples"])
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    samples = []
    prev = None
    t = 0.0
    while t < dur:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        box = None
        if roi_first:
            # search region: clamp(8*prev_h, 480, H) centered on prediction
            if prev is not None:
                size = min(max(prev["h"] * 8, 480), H)
                cxp, cyp = prev["cx"], prev["cy"]
            else:
                # seed: full-frame best person (single-rider clip)
                full = detect_persons(frame)
                prev = full[0] if full else None
                samples.append({"t": t, "box": prev})
                t += SAMPLE_INTERVAL
                continue
            x0 = min(max(cxp - size / 2, 0), W - size)
            y0 = min(max(cyp - size / 2, 0), max(0, H - size))
            dets = detect_persons(frame, (x0, y0, min(size, W), min(size, H)))
        else:
            dets = detect_persons(frame)
        # nearest-to-prediction gate (simplified tracker: single rider)
        if dets and prev is not None:
            gate = 0.3 * np.hypot(W, H)
            best = min(dets, key=lambda d: np.hypot(d["cx"] - prev["cx"], d["cy"] - prev["cy"]))
            if np.hypot(best["cx"] - prev["cx"], best["cy"] - prev["cy"]) < gate:
                box = best
        if box is None and not roi_first:
            pass
        if box is None:
            full = detect_persons(frame)
            if full and prev is not None:
                best = min(full, key=lambda d: np.hypot(d["cx"] - prev["cx"], d["cy"] - prev["cy"]))
                if np.hypot(best["cx"] - prev["cx"], best["cy"] - prev["cy"]) < 0.3 * np.hypot(W, H):
                    box = best
        if box is not None:
            prev = box
        samples.append({"t": t, "box": box})
        t += SAMPLE_INTERVAL
    cap.release()
    np.savez(key, samples=np.array(samples, dtype=object))
    return samples


# ---------- camera path (ports stabilize.ts) ----------


def cam_pass(path, rider_at):
    key = os.path.join(CACHE, f"cam_{os.path.basename(path)}.npz")
    if os.path.exists(key):
        z = np.load(key)
        return {k: z[k] for k in z.files}
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    AW, AH = 1024, 512  # analysis grab
    LW = 256  # local window (grab px)
    win_full = cv2.createHanningWindow((AW, AH), cv2.CV_32F)
    win_half = cv2.createHanningWindow((AW // 2, AH), cv2.CV_32F)
    win_loc = cv2.createHanningWindow((LW, LW), cv2.CV_32F)
    prev = None
    ts, fx, fy, fpk, lx, ly, lpk, rolls = [], [], [], [], [], [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        g = cv2.cvtColor(cv2.resize(frame, (AW, AH)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            (dx, dy), pk = cv2.phaseCorrelate(prev, g, win_full)
            # local window around rider
            rx, ry = rider_at(t)
            x0 = int(min(max(rx / W * AW - LW / 2, 0), AW - LW))
            y0 = int(min(max(ry / H * AH - LW / 2, 0), AH - LW))
            (ldx, ldy), lp = cv2.phaseCorrelate(
                prev[y0 : y0 + LW, x0 : x0 + LW], g[y0 : y0 + LW, x0 : x0 + LW], win_loc
            )
            # roll from left/right half dy differential
            (l_dx, l_dy), _ = cv2.phaseCorrelate(prev[:, : AW // 2], g[:, : AW // 2], win_half)
            (r_dx, r_dy), _ = cv2.phaseCorrelate(prev[:, AW // 2 :], g[:, AW // 2 :], win_half)
            roll = ((r_dy - l_dy) * (H / AH)) / (W / 2)
            ts.append(t)
            fx.append(dx * W / AW)
            fy.append(dy * H / AH)
            fpk.append(pk)
            lx.append(ldx * W / AW)
            ly.append(ldy * H / AH)
            lpk.append(lp)
            rolls.append(roll if abs(roll) < 0.05 else 0.0)
        prev = g
        i += 1
    cap.release()
    out = {
        "ts": np.array(ts),
        "fx": np.array(fx),
        "fy": np.array(fy),
        "fpk": np.array(fpk),
        "lx": np.array(lx),
        "ly": np.array(ly),
        "lpk": np.array(lpk),
        "roll": np.array(rolls),
        "fps": np.array([fps]),
        "size": np.array([W, H]),
    }
    np.savez(key, **out)
    return out


# ---------- path build (ports path.ts) ----------


def fill_gaps(vals):
    v = np.array(vals, dtype=np.float64)
    idx = np.where(~np.isnan(v))[0]
    if len(idx) == 0:
        return v
    v = np.interp(np.arange(len(v)), idx, v[idx])
    return v


def median_filter(v, r=2):
    out = v.copy()
    for i in range(len(v)):
        out[i] = np.median(v[max(0, i - r) : i + r + 1])
    return out


def gaussian_smooth(v, sigma):
    if sigma <= 0:
        return v.copy()
    r = max(1, int(np.ceil(sigma * 3)))
    k = np.exp(-(np.arange(-r, r + 1) ** 2) / (2 * sigma * sigma))
    k /= k.sum()
    return np.convolve(np.pad(v, r, mode="edge"), k, mode="valid")


def build_path(samples, cam_t, cam_x, cam_y, W, H, smooth_sec=SMOOTH_SEC, pad=PAD_FACTOR):
    ts = np.array([s["t"] for s in samples])
    cx = fill_gaps([s["box"]["cx"] if s["box"] else np.nan for s in samples])
    cy = fill_gaps([s["box"]["cy"] if s["box"] else np.nan for s in samples])
    ph = fill_gaps([s["box"]["h"] if s["box"] else np.nan for s in samples])
    # world coords
    camx_s = np.interp(ts, cam_t, cam_x)
    camy_s = np.interp(ts, cam_t, cam_y)
    cx = median_filter(cx) - camx_s
    cy = median_filter(cy) - camy_s
    ph = median_filter(ph)
    sigma = smooth_sec / SAMPLE_INTERVAL
    sph = gaussian_smooth(ph, sigma * 2)
    crop_h = np.clip(sph * pad, H / MAX_ZOOM, H)
    crop_w = crop_h * (W / H)
    lim_x = crop_w * LEASH_X / 2
    lim_y = crop_h * LEASH_Y / 2
    # smoothed cam for bounds
    fps_cam = 1 / np.median(np.diff(cam_t))
    smx = np.interp(ts, cam_t, gaussian_smooth(cam_x, smooth_sec * fps_cam))
    smy = np.interp(ts, cam_t, gaussian_smooth(cam_y, smooth_sec * fps_cam))
    mX, mY = 0.02 * W, 0.025 * H

    def constrain(sx, sy):
        fxx = sx + smx
        fyy = sy + smy
        lo_x, hi_x = crop_w / 2 + mX, W - crop_w / 2 - mX
        lo_y, hi_y = crop_h / 2 + mY, H - crop_h / 2 - mY
        fxx = np.where(hi_x < lo_x, W / 2, np.clip(fxx, lo_x, hi_x))
        fyy = np.where(hi_y < lo_y, H / 2, np.clip(fyy, lo_y, hi_y))
        return fxx - smx, fyy - smy

    sx = gaussian_smooth(cx, sigma)
    sy = gaussian_smooth(cy, sigma)
    for it in range(6):
        s = sigma * 0.55 ** (it + 1)
        sx = np.clip(sx, cx - lim_x, cx + lim_x)
        sy = np.clip(sy, cy - lim_y, cy + lim_y)
        sx, sy = constrain(sx, sy)
        sx = gaussian_smooth(sx, s)
        sy = gaussian_smooth(sy, s)
    sx = np.clip(sx, cx - lim_x, cx + lim_x)
    sy = np.clip(sy, cy - lim_y, cy + lim_y)
    sx, sy = constrain(sx, sy)
    return {"ts": ts, "cx": sx, "cy": sy, "cropW": crop_w, "cropH": crop_h}


# ---------- render ----------


def render(path, out_path, crop_path, cam_t, cam_x, cam_y, roll_cum, smooth_sec=SMOOTH_SEC):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    outW, outH = 1920, 1080
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (outW, outH))
    fps_cam = 1 / np.median(np.diff(cam_t))
    sroll = gaussian_smooth(roll_cum, smooth_sec * fps_cam)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        # step lookup for cam (per-frame), interp for path (slow)
        ci = min(np.searchsorted(cam_t, t, "right") - 1, len(cam_t) - 1)
        ci = max(ci, 0)
        camx, camy, camr = cam_x[ci], cam_y[ci], roll_cum[ci]
        roll_res = camr - sroll[ci]
        pcx = np.interp(t, crop_path["ts"], crop_path["cx"]) + camx
        pcy = np.interp(t, crop_path["ts"], crop_path["cy"]) + camy
        cw = np.interp(t, crop_path["ts"], crop_path["cropW"])
        chh = np.interp(t, crop_path["ts"], crop_path["cropH"])
        pcx = min(max(pcx, cw / 2), W - cw / 2)
        pcy = min(max(pcy, chh / 2), H - chh / 2)
        k = outW / cw
        # output = scale k, rotate -roll_res about crop center, translate
        M = cv2.getRotationMatrix2D((float(pcx), float(pcy)), np.degrees(roll_res), k)
        M[0, 2] += outW / 2 - pcx
        M[1, 2] += outH / 2 - pcy
        vw.write(cv2.warpAffine(frame, M, (outW, outH), flags=cv2.INTER_LINEAR))
        i += 1
    cap.release()
    vw.release()


# ---------- main ----------

if __name__ == "__main__":
    clip = sys.argv[1]
    samples = detect_pass(clip)
    n_det = sum(1 for s in samples if s["box"])
    hs = [s["box"]["h"] for s in samples if s["box"]]
    print(f"detections: {n_det}/{len(samples)} samples, rider h median {np.median(hs):.0f}px")

    def rider_at(t):
        ts = np.array([s["t"] for s in samples])
        cx = fill_gaps([s["box"]["cx"] if s["box"] else np.nan for s in samples])
        cy = fill_gaps([s["box"]["cy"] if s["box"] else np.nan for s in samples])
        return np.interp(t, ts, cx), np.interp(t, ts, cy)

    cam = cam_pass(clip, rider_at)
    W, H = int(cam["size"][0]), int(cam["size"][1])
    print(f"cam frames: {len(cam['ts'])}, full peak mean {cam['fpk'].mean():.3f}, local peak mean {cam['lpk'].mean():.3f}")

    # translation choice experiment: full vs local vs fused
    for name, dx, dy in [
        ("full", cam["fx"], cam["fy"]),
        ("local", cam["lx"], cam["ly"]),
    ]:
        cum_x, cum_y = np.cumsum(dx), np.cumsum(dy)
        hp = lambda v: v - gaussian_smooth(v, 15)
        print(f"  {name}: HF translation rms x={hp(cum_x).std():.2f} y={hp(cum_y).std():.2f}")

    # default: full-frame translation (top-down clips), cumulative
    cum_x, cum_y = np.cumsum(cam["fx"]), np.cumsum(cam["fy"])
    roll_cum = np.cumsum(cam["roll"])
    cpath = build_path(samples, cam["ts"], cum_x, cum_y, W, H)
    if "--render" in sys.argv:
        out = sys.argv[sys.argv.index("--render") + 1]
        render(clip, out, cpath, cam["ts"], cum_x, cum_y, roll_cum)
        print("rendered", out)
