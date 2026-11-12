"""Pass detector and frame picker for a static camera.

Idea: the camera does not move, so anything that changes is the vehicle.
A background-subtraction model gives a 'moving stuff' mask per frame.
A pass is 'active' while a big moving object is in view.
The best frame is the one where the object is largest, fully inside the picture and sharp.
"""
import cv2
import numpy as np


class PassDetector:
    def __init__(self, size, scale=0.25, min_height_frac=0.35, margin_frac=0.02):
        self.scale = scale
        self.bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
        self.min_h = min_height_frac
        self.margin = margin_frac
        self.size = size
        self.candidates = []  # (score, idx, t, bbox)

    def update(self, idx, t, frame):
        small = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        fg = self.bg.apply(small)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats((fg > 0).astype(np.uint8))
        if n < 2:
            return None
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h, area = stats[k]
        Hs, Ws = small.shape[:2]
        if h / Hs < self.min_h:
            return None
        m = self.margin * Ws
        inside = x > m and (x + w) < Ws - m          # not cut off left/right
        gray = cv2.cvtColor(small[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        bbox = tuple(int(v / self.scale) for v in (x, y, w, h))
        ev = dict(idx=idx, t=t, bbox=bbox, inside=bool(inside), sharp=float(sharp), area=int(area / self.scale ** 2))
        self.candidates.append(ev)
        return ev

    def best(self, k=3, min_gap=10):
        c = [e for e in self.candidates if e["inside"]]
        if not c:
            return []
        amax = max(e["area"] for e in c); smax = max(e["sharp"] for e in c)
        for e in c:
            e["score"] = 0.7 * e["area"] / amax + 0.3 * e["sharp"] / smax
        c.sort(key=lambda e: -e["score"])
        out = []
        for e in c:
            if all(abs(e["idx"] - o["idx"]) >= min_gap for o in out):
                out.append(e)
            if len(out) == k:
                break
        return out
