"""Full DroneZoom pipeline as a callable: clip in, stabilized auto-zoomed
H.264 MP4 out. Wraps the validated analysis/v4 components."""

import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))

from pipeline import detect_pass, gaussian_smooth  # noqa: E402
from v3 import apply_a, pass_a  # noqa: E402
from v4 import (  # noqa: E402
    build_path_v4,
    decompose,
    residuals_incremental,
    to_source_Ms,
    track_blobs,
)

PAD_FACTOR = 5.0
SMOOTH_SEC = 1.0


def render_ffmpeg(clip, out_path, path, Ds, fps, W, H):
    import imageio_ffmpeg

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    outW, outH = 1920, 1080
    proc = subprocess.Popen(
        [
            exe, "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{outW}x{outH}",
            "-r", f"{fps:.4f}", "-i", "-",
            "-c:v", "libx264", "-preset", "medium", "-crf", "19",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            out_path,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    th_d, sc_d, _, _ = decompose(Ds)
    cap = cv2.VideoCapture(clip)
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
        out = cv2.warpAffine(frame, M, (outW, outH), flags=cv2.INTER_LINEAR)
        proc.stdin.write(out.tobytes())
        i += 1
    cap.release()
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg encode failed")


def process(clip_path, out_path, log=print):
    log("  camera pass (grid phase correlation)…")
    A = pass_a(clip_path)
    W, H = int(A["size"][0]), int(A["size"][1])
    fps = float(A["fps"][0])
    Ms_src = to_source_Ms(A["Ms"], W, H)
    Ds = residuals_incremental(Ms_src, fps, SMOOTH_SEC)

    log("  rider detection (YOLO seed + anchors)…")
    yolo = detect_pass(clip_path)
    n_det = sum(1 for s in yolo if s["box"])
    if n_det == 0:
        raise RuntimeError("no rider detected anywhere in the clip — cannot seed tracking")

    log("  motion-blob tracking…")
    track = track_blobs(clip_path, A, yolo)
    found_pct = float(track["found"].mean() * 100)

    # zoom-out-on-pan
    speeds = np.hypot(Ms_src[:, 0, 2], Ms_src[:, 1, 2])
    r = int(fps * 0.3)
    env = np.array([speeds[max(0, i - r) : i + r + 1].max() for i in range(len(speeds))])
    sp_s = gaussian_smooth(env, fps * 0.3)
    lo, hi = 0.021 * W, 0.052 * W
    x01 = np.clip((sp_s - lo) / (hi - lo), 0, 1)
    wfac = 1 + 0.8 * (x01 * x01 * (3 - 2 * x01))
    widen = (np.arange(len(wfac)) / fps, wfac)

    log("  building camera path…")
    path = build_path_v4(track, Ds, fps, W, H, pad=PAD_FACTOR, widen=widen)

    log("  rendering + H.264 encode…")
    render_ffmpeg(clip_path, out_path, path, Ds, fps, W, H)
    return {
        "fps": fps,
        "size": (W, H),
        "yolo_samples": n_det,
        "blob_found_pct": found_pct,
        "out_bytes": os.path.getsize(out_path),
    }


if __name__ == "__main__":
    stats = process(sys.argv[1], sys.argv[2])
    print(stats)
