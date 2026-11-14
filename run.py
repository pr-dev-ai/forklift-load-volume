"""Command line version. For the UI run:  python app.py

Usage:  python run.py [video_or_rtsp_url] [--realtime]
Reads the feed frame by frame, detects the pass, picks the best frames,
finds the corners of the load (automatically, and from the hand-marked file for comparison),
fits a 3D box and writes out/result.json + out/overlay_*.jpg
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from stream import FrameSource          # noqa: E402
from detect import PassDetector         # noqa: E402
from boxfit import ORDER                # noqa: E402
from measure import measure             # noqa: E402
from autocorners import auto_corners    # noqa: E402
from explain import step_images         # noqa: E402


def detect_pass(uri, realtime=False, keep=()):
    src = FrameSource(uri, realtime=realtime)
    det = PassDetector(src.size)
    frames, t_first, t_last, n = {}, None, None, 0
    for idx, t, frame in src:
        ev = det.update(idx, t, frame)
        if ev and idx > 5:
            t_first = t if t_first is None else t_first
            t_last = t
        if idx in keep:
            frames[idx] = frame.copy()
        n = idx + 1
    picks = det.best(k=3)
    info = dict(source=Path(str(uri)).name if "://" not in str(uri) else str(uri), fps=src.fps, frames=n, size=list(src.size),
                pass_window_s=[round(t_first, 2), round(t_last, 2)],
                auto_picked_frames=[p["idx"] for p in picks])
    return info, frames


def draw(img, corners, reproj, res, label):
    b, u = res["box_fit"], res.get("uncertainty", {})
    rp = np.array(reproj).astype(int)
    for a, c in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (4, 5), (5, 3)]:
        cv2.line(img, tuple(rp[a]), tuple(rp[c]), (0, 255, 0), 3)
    for k in ORDER:
        cv2.circle(img, tuple(int(v) for v in corners[k]), 7, (0, 255, 255), -1)
    txt = (f"{label}: {b['width_m']:.1f} x {b['depth_m']:.2f} x {b['height_m']:.2f} m = {b['volume_m3']:.1f} m3"
           f"  (90% range {u.get('volume_m3_p05', '?')}-{u.get('volume_m3_p95', '?')})")
    cv2.rectangle(img, (20, 20), (1420, 80), (0, 0, 0), -1)
    cv2.putText(img, txt, (30, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return img


def main():
    uri = next((a for a in sys.argv[1:] if not a.startswith("--")), str(ROOT / "data" / "pass_0216.mp4"))
    ann = json.load(open(ROOT / "annotations" / "corners.json"))
    hand = {k: v for k, v in ann.items() if k.isdigit()}
    idxs = sorted(int(k) for k in hand)
    info, frames = detect_pass(uri, "--realtime" in sys.argv, set(idxs))
    size = tuple(info["size"])

    auto = auto_corners(uri, idxs)                      # same frames, no clicking
    results = {"hand_marked": measure(hand, size), "automatic": measure(auto, size)}
    dist = [float(np.hypot(auto[k][p][0] - hand[k][p][0], auto[k][p][1] - hand[k][p][1])) for k in hand for p in ORDER]

    (ROOT / "out").mkdir(exist_ok=True)
    for name, corners in (("hand_marked", hand), ("automatic", auto)):
        reproj = results[name].pop("reproj")
        for i in idxs:
            tag = "" if name == "hand_marked" else "auto_"
            cv2.imwrite(str(ROOT / "out" / f"overlay_{tag}{i:04d}.jpg"),
                        draw(frames[i].copy(), corners[str(i)], reproj[str(i)], results[name], name.replace("_", " ")))
    for n, img in enumerate(step_images(uri, idxs[-1], auto[str(idxs[-1])]), 1):   # what each step saw
        cv2.imwrite(str(ROOT / "out" / f"step_{n}.jpg"), cv2.resize(img, (1280, 720)), [cv2.IMWRITE_JPEG_QUALITY, 88])
    res = {**info, **results, "automatic_corners": auto,
           "corner_distance_px": dict(mean=round(float(np.mean(dist)), 1), max=round(float(np.max(dist)), 1))}
    json.dump(res, open(ROOT / "out" / "result.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
