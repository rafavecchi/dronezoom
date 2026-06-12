"""Per-segment track diagnostics on a source clip: blob-found %, blob vs
YOLO error, YOLO score — find when/why the track diverges."""

import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from pipeline import detect_pass, fill_gaps  # noqa: E402
from v3 import pass_a  # noqa: E402
from v4 import track_blobs  # noqa: E402

clip = sys.argv[1]
A = pass_a(clip)
fps = float(A["fps"][0])
W, H = int(A["size"][0]), int(A["size"][1])
yolo = detect_pass(clip)
track = track_blobs(clip, A, yolo)

ts = track["ts"]
dur = ts[-1]
print(f"{clip}: {dur:.0f}s, blob found {track['found'].mean() * 100:.0f}% overall")
yts = np.array([s["t"] for s in yolo])
yscore = np.array([s["box"]["score"] if s["box"] else 0.0 for s in yolo])
for lo in range(0, int(dur), 5):
    m = (ts >= lo) & (ts < lo + 5)
    found = track["found"][m].mean() * 100
    # blob vs YOLO where YOLO confident in this window
    errs = []
    for s in yolo:
        if lo <= s["t"] < lo + 5 and s["box"] and s["box"]["score"] > 0.35:
            mx = np.interp(s["t"], ts, track["x"])
            my = np.interp(s["t"], ts, track["y"])
            errs.append(np.hypot(mx - s["box"]["cx"], my - s["box"]["cy"]))
    e = f"err med {np.median(errs):4.0f}px ({len(errs)})" if errs else "no conf YOLO"
    ym = (yts >= lo) & (yts < lo + 5)
    print(f"  t={lo:2d}-{lo + 5:2d}s: blob {found:3.0f}%  {e}  yolo-best {yscore[ym].max():.2f}")
