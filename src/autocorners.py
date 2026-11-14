"""Find the six corners of the load automatically (no clicking).

What is detected from pixels, per frame:
  1. Silhouette of everything that moved: |frame - background|, background = median of the clip.
  2. Front face of the load: crate-blue pixels inside the silhouette. A line is fitted to each
     of its four sides (RANSAC), the four intersections are corners A, B, C, D.
  3. Far edge of the side face: walking left from the edge A-D, the column where the dark
     crate wall ends and the brighter truck cab begins.

  4. Corners E and F: at that far edge, the bottom of the dark wall (bright pallet wood below)
     and its top.

The result has the same six points a person would click, so the box fit is unchanged.

Tuned for this clip (blue crates, cab brighter than the crate wall). See README, "Where it fails".
"""
import cv2
import numpy as np

S = 0.5                      # work at half resolution
BLUE_H = (102, 130)          # OpenCV hue range of the crates
BLUE_S_MIN, BLUE_V_MIN = 45, 25
DIFF_THR = 35                # silhouette threshold on |frame - background|
WOOD_H = (9, 24)             # hue range of sunlit pallet wood
WOOD_S_MIN, WOOD_V_MIN = 90, 110
WOOD_BLOCK_S = 125           # a pallet block's median saturation is at least this
AUTO_SIGMA_PX = 8            # assumed error of an automatic corner, for the range
MAX_BLOCK_GAP = 0.35         # widest gap between pallet blocks, as a share of the load's height in px


def small(frame):
    return cv2.resize(frame, None, fx=S, fy=S, interpolation=cv2.INTER_AREA)


def load(uri, idxs, step=6):
    """One pass over the clip: the background, and the full frames we want to measure.

    Background = median over time of every `step`-th frame. The truck covers each pixel only
    part of the time, so it drops out. (On a live camera this is the running background model.)
    """
    cap = cv2.VideoCapture(str(uri))
    smalls, frames, i = [], {}, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            smalls.append(small(f))
        if i in idxs:
            frames[i] = f
        i += 1
    cap.release()
    return np.median(np.stack(smalls), 0).astype(np.uint8), frames


def silhouette(sm, bg):
    m = (cv2.absdiff(sm, bg).max(2) > DIFF_THR).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))


def front_mask(sm, sil):
    h, s, v = cv2.split(cv2.cvtColor(sm, cv2.COLOR_BGR2HSV))
    blue = ((h >= BLUE_H[0]) & (h <= BLUE_H[1]) & (s >= BLUE_S_MIN) & (v >= BLUE_V_MIN)).astype(np.uint8) * 255
    m = blue & sil
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        raise RuntimeError("no crate-coloured region found")
    return lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))


def _ransac_line(t, u, outer, thr=2.5, iters=400, max_beyond=0.2, seed=0):
    """Fit u = a*t + b to the OUTER boundary of a mask side.

    Holes in the mask pull boundary points inwards, so the true edge is the outermost line that
    still has support. outer = +1 if "outside" means larger u, -1 if smaller. A candidate line
    is rejected when more than max_beyond of the points lie outside it (those would be leaks).
    """
    rng = np.random.default_rng(seed)
    best, best_in = None, None
    for _ in range(iters):
        i, j = rng.choice(len(t), 2, replace=False)
        if t[i] == t[j]:
            continue
        a = (u[j] - u[i]) / (t[j] - t[i]); b = u[i] - a * t[i]
        r = (u - (a * t + b)) * outer
        if (r > thr).mean() > max_beyond:
            continue
        inl = np.abs(r) < thr
        if best_in is None or inl.sum() > best_in.sum():
            best, best_in = (a, b), inl
    if best is None:
        raise RuntimeError("no boundary line found")
    a, b = np.polyfit(t[best_in], u[best_in], 1)
    return float(a), float(b), float(best_in.mean())


def front_quad(mask):
    """Four corners of the front face (half-res pixels) from its mask: A, B, C, D."""
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    rows = np.arange(int(y0 + 0.1 * (y1 - y0)), int(y1 - 0.1 * (y1 - y0)))
    cols = np.arange(int(x0 + 0.1 * (x1 - x0)), int(x1 - 0.1 * (x1 - x0)))
    rows = rows[mask[rows].any(1)]; cols = cols[mask[:, cols].any(0)]
    left = np.array([np.argmax(mask[r]) for r in rows], float)
    right = np.array([mask.shape[1] - 1 - np.argmax(mask[r, ::-1]) for r in rows], float)
    top = np.array([np.argmax(mask[:, c]) for c in cols], float)
    bot = np.array([mask.shape[0] - 1 - np.argmax(mask[::-1, c]) for c in cols], float)
    L = _ransac_line(rows.astype(float), left, -1)   # x = a*y + b
    R = _ransac_line(rows.astype(float), right, +1)
    T = _ransac_line(cols.astype(float), top, -1)    # y = a*x + b
    B = _ransac_line(cols.astype(float), bot, +1)

    def cross(v, h):                                # x = av*y + bv  with  y = ah*x + bh
        x = (v[0] * h[1] + v[1]) / (1 - v[0] * h[0])
        return [x, h[0] * x + h[1]]
    quad = dict(A=cross(L, B), B=cross(R, B), C=cross(R, T), D=cross(L, T))
    support = min(L[2], R[2], T[2], B[2])            # share of boundary points on the fitted lines
    return quad, support, dict(L=L, R=R, T=T, B=B, cols=cols.astype(float), bot=bot)


def side_bottom(sm, quad):
    """Bottom edge of the side wall and where it ends, from the sunlit pallet wood under it.

    The pallets carry the load, so the top of the wood strip is the edge A-E and the strip's
    far end is corner E. Returns (a, b) of y = a*x + b and far_x, in half-res pixels.
    """
    A, D, B = quad["A"], quad["D"], quad["B"]
    hgt, wid = A[1] - D[1], B[0] - A[0]
    x0, x1 = max(0, int(A[0] - 0.9 * wid)), int(A[0]) + 4
    y0, y1 = int(A[1] - 0.25 * hgt), min(sm.shape[0], int(A[1] + 0.2 * hgt))
    h, s, v = cv2.split(cv2.cvtColor(sm[y0:y1, x0:x1], cv2.COLOR_BGR2HSV))
    wood = ((h >= WOOD_H[0]) & (h <= WOOD_H[1]) & (s >= WOOD_S_MIN) & (v >= WOOD_V_MIN)).astype(np.uint8)
    wood = cv2.morphologyEx(wood, cv2.MORPH_CLOSE, np.ones((5, 21), np.uint8))   # bridge gaps between blocks
    wood = cv2.morphologyEx(wood, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(wood)
    if n < 2:
        raise RuntimeError("no pallet wood found under the side wall")
    blocks = [j for j in range(1, n)
              if st[j, cv2.CC_STAT_AREA] >= 30 and np.median(s[lab == j]) >= WOOD_BLOCK_S]   # not speckle, not paler than wood
    if not blocks:
        raise RuntimeError("no pallet blocks found under the side wall")
    # top pixel of every block column; the edge A-E is the line through the block nearest
    # corner A that the most of these points agree with
    tops = {}
    for j in blocks:
        m = lab == j
        cols = np.nonzero(m.any(0))[0]
        tops[j] = (cols.astype(float), np.argmax(m[:, cols], 0).astype(float))
    px = np.concatenate([t[0] for t in tops.values()]); py = np.concatenate([t[1] for t in tops.values()])
    near = px > px.max() - 15
    ax, ay = px[near].mean(), np.median(py[near])
    slopes = np.linspace(-0.7, 0.3, 201)
    votes = [(np.abs(ay + m_ * (px - ax) - py) < 3).sum() for m_ in slopes]
    m0 = slopes[int(np.argmax(votes))]
    inl = np.abs(ay + m0 * (px - ax) - py) < 4
    a, b = np.polyfit(px[inl], py[inl], 1)
    strip = np.zeros(wood.shape, bool)
    for j, (cx, cy) in tops.items():                       # keep the blocks whose top runs along that line
        if np.median(np.abs(a * cx + b - cy)) < 5:
            strip |= lab == j
    # walk left from corner A over the blocks; a wide gap means the pallet has ended
    cols = np.nonzero(strip.any(0))[0][::-1]
    far = cols[0]
    for c in cols:
        if far - c > MAX_BLOCK_GAP * hgt:
            break
        far = c
    return (a, b + y0 - a * x0), float(far + x0)


def side_top(sm, A, D, bottom, far_x, layers):
    """Corner F, from the direction in which the crate layers run along the side wall.

    The top edge D-F is hidden behind the mast for most of its length, but the layer joints of
    the crates are visible all over the wall. They are parallel in 3D, so in the image they all
    aim at one vanishing point, and that point lies on the bottom edge A-E extended. Try
    positions for it, and keep the one where lines from the layer joints on edge A-D run along
    the strongest horizontal edges. The line from D to that point is the top edge.
    """
    g = cv2.GaussianBlur(cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY), (0, 0), 1.0).astype(np.float32)
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    A, D = np.array(A, float), np.array(D, float)
    u = np.array([1.0, bottom[0]]) / np.hypot(1.0, bottom[0])
    xs = np.arange(far_x + 4, A[0] - 4, 1.0)
    best = (-1.0, None)
    for dist in 1.0 / np.linspace(1 / 300.0, 1 / 20000.0, 200):       # distance of the point from A
        vz = A - u * dist
        score = 0.0
        for k in range(1, layers + 1):
            p = A + (D - A) * k / layers
            ys = p[1] + (xs - p[0]) * (vz[1] - p[1]) / (vz[0] - p[0])
            score += gy[np.clip(ys.round().astype(int), 0, gy.shape[0] - 1), xs.astype(int)].mean()
        if score > best[0]:
            best = (score, vz)
    vz = best[1]
    return [far_x, D[1] + (far_x - D[0]) * (vz[1] - D[1]) / (vz[0] - D[0])]


def _bottom_right(lines, A):
    """B: on the right edge, on the line from A along the lowest supported bottom points."""
    cols, bot, R = lines["cols"], lines["bot"], lines["R"]
    far = cols > A[0] + 0.5 * (cols.max() - A[0])          # right half: clean crate bottoms
    slope = float(np.percentile((bot[far] - A[1]) / (cols[far] - A[0]), 80))
    h = (slope, A[1] - slope * A[0])
    x = (R[0] * h[1] + R[1]) / (1 - R[0] * h[0])
    return [x, h[0] * x + h[1]]


def auto_corners(uri, idxs, layers=5):
    """Corners for the given frame numbers, in the same format as annotations/corners.json."""
    bg, frames = load(uri, set(idxs))
    out = {}
    for i in idxs:
        sm = small(frames[i])
        quad, support, lines = front_quad(front_mask(sm, silhouette(sm, bg)))
        (a, b), far_x = side_bottom(sm, quad)
        L = lines["L"]                                     # x = L0*y + L1, the edge A-D
        ya = (a * L[1] + b) / (1 - a * L[0])               # A = side bottom edge meets edge A-D
        quad["A"] = [L[0] * ya + L[1], ya]
        quad["B"] = _bottom_right(lines, quad["A"])
        quad["E"] = [far_x, a * far_x + b]
        quad["F"] = side_top(sm, quad["A"], quad["D"], (a, b), far_x, layers)
        c = {k: [round(float(quad[k][0] / S)), round(float(quad[k][1] / S))] for k in "ABCDEF"}
        c["sigma_px"] = AUTO_SIGMA_PX
        out[str(i)] = c
    return out
