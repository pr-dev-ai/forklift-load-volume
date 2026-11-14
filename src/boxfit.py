"""Fit a 3D box of known width to 6 image corners (single view, unknown camera).

Box frame: origin at the near bottom corner A.
  X = along the front face (known width W), Y = up (height H), Z = along the side face (depth D).
Corners:  A(0,0,0) B(W,0,0) C(W,H,0) Dc(0,H,0) E(0,0,D) F(0,H,D)
Unknowns: focal length f, camera rotation (3), translation (3), H, D  -> 9 unknowns, 12 equations.
"""
import cv2
import numpy as np


def _lm(fun, x0, iters=200):
    """Small Levenberg-Marquardt solver (numeric Jacobian) so we only need numpy + opencv."""
    x = np.array(x0, float); r = fun(x); cost = r @ r; lam = 1e-2
    for _ in range(iters):
        J = np.empty((r.size, x.size))
        for j in range(x.size):
            h = 1e-6 * max(1.0, abs(x[j])); xp = x.copy(); xp[j] += h
            J[:, j] = (fun(xp) - r) / h
        A = J.T @ J; g = J.T @ r
        improved = False
        for _ in range(12):
            try:
                step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -g)
            except np.linalg.LinAlgError:
                lam *= 10; continue
            rn = fun(x + step); cn = rn @ rn
            if np.isfinite(cn) and cn < cost:
                x, r, improved = x + step, rn, True
                done = (cost - cn) < 1e-10 * max(cost, 1e-12)
                cost = cn; lam = max(lam / 5, 1e-9)
                break
            lam *= 5
        if not improved or done:
            break
    return x, cost

ORDER = ["A", "B", "C", "D", "E", "F"]


def model_points(W, H, D):
    return np.array([[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0], [0, 0, D], [0, H, D]], float)


def project(params, W, cx, cy):
    logf, rx, ry, rz, tx, ty, tz, H, D = params
    f = np.exp(logf)
    P = model_points(W, H, D)
    # image y points down, box Y points up -> flip Y
    P = P * np.array([1, -1, 1])
    R, _ = cv2.Rodrigues(np.array([rx, ry, rz], float))
    Pc = P @ R.T + np.array([tx, ty, tz])
    u = f * Pc[:, 0] / Pc[:, 2] + cx
    v = f * Pc[:, 1] / Pc[:, 2] + cy
    return np.stack([u, v], 1), Pc


def fit_box(corners_px, W, img_size, n_starts=40, seed=0):
    """corners_px: dict A..F -> (x, y). Returns dict with H, D, f, rms."""
    cx, cy = img_size[0] / 2, img_size[1] / 2
    obs = np.array([corners_px[k] for k in ORDER], float)
    rng = np.random.default_rng(seed)

    def resid(p):
        uv, Pc = project(p, W, cx, cy)
        r = (uv - obs).ravel()
        # keep the box in front of the camera
        pen = np.clip(0.1 - Pc[:, 2], 0, None) * 1e3
        return np.concatenate([r, pen])

    best = None
    for _ in range(n_starts):
        p0 = [np.log(rng.uniform(800, 6000)), *rng.normal(0, 0.4, 3),
              rng.uniform(-2, 2), rng.uniform(-1, 2), rng.uniform(4, 25),
              rng.uniform(0.8, 2.5), rng.uniform(0.8, 4.0)]
        x, cost = _lm(resid, p0)
        H, D = x[7], x[8]
        if not np.isfinite(cost) or H <= 0 or D <= 0:
            continue
        if best is None or cost < best[1]:
            best = (x, cost)
    bx = best[0]
    uv, _ = project(bx, W, cx, cy)
    rms = float(np.sqrt(np.mean((uv - obs) ** 2)))
    return dict(H=float(bx[7]), D=float(bx[8]), f=float(np.exp(bx[0])),
                rms_px=rms, params=bx.tolist(), reproj=uv.tolist())


def monte_carlo(corners_px, W, img_size, sigma_px=4.0, w_sigma=0.03, n=200, seed=1):
    """Jitter the clicked corners and the assumed width to get an honest spread."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        c = {k: (v[0] + rng.normal(0, sigma_px), v[1] + rng.normal(0, sigma_px)) for k, v in corners_px.items()}
        Wj = W * (1 + rng.normal(0, w_sigma))
        r = fit_box(c, Wj, img_size, n_starts=8, seed=i)
        if r["rms_px"] < 6:
            out.append((Wj, r["H"], r["D"], Wj * r["H"] * r["D"]))
    return np.array(out)


def fit_joint(frames, W, img_size, n_starts=30, seed=0):
    """Fit ONE box (shared H, D) and ONE focal length to several frames at once.

    frames: list of corner dicts (one per frame). Each frame gets its own pose
    because the truck has moved. Sharing H, D and f makes the answer much steadier
    than fitting each frame alone.
    """
    cx, cy = img_size[0] / 2, img_size[1] / 2
    obs = [np.array([c[k] for k in ORDER], float) for c in frames]
    n = len(frames)
    rng = np.random.default_rng(seed)

    def unpack(p, i):
        return [p[0], *p[3 + 6 * i: 9 + 6 * i], p[1], p[2]]

    def resid(p):
        out = []
        for i in range(n):
            uv, Pc = project(unpack(p, i), W, cx, cy)
            out.append((uv - obs[i]).ravel())
            out.append(np.clip(0.1 - Pc[:, 2], 0, None) * 1e3)
        return np.concatenate(out)

    best = None
    for _ in range(n_starts):
        p0 = [np.log(rng.uniform(800, 6000)), rng.uniform(0.8, 2.5), rng.uniform(0.8, 4.0)]
        for i in range(n):
            p0 += [*rng.normal(0, 0.4, 3), rng.uniform(-2, 2), rng.uniform(-1, 2), rng.uniform(4, 25)]
        x, cost = _lm(resid, p0)
        if not np.isfinite(cost) or x[1] <= 0 or x[2] <= 0:
            continue
        if best is None or cost < best[1]:
            best = (x, cost)
    x = best[0]
    reproj = [project(unpack(x, i), W, cx, cy)[0] for i in range(n)]
    rms = float(np.sqrt(np.mean([(reproj[i] - obs[i]) ** 2 for i in range(n)])))
    return dict(H=float(x[1]), D=float(x[2]), f=float(np.exp(x[0])), rms_px=rms,
                reproj=[r.tolist() for r in reproj])


def monte_carlo_joint(frames, sigmas, W, img_size, w_sigma=0.02, n=150, seed=1):
    """Shake the clicked corners (by their click error) and the assumed width, refit, collect volumes."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        fr = [{k: (v[0] + rng.normal(0, s), v[1] + rng.normal(0, s)) for k, v in c.items()}
              for c, s in zip(frames, sigmas)]
        Wj = W * (1 + rng.normal(0, w_sigma))
        r = fit_joint(fr, Wj, img_size, n_starts=6, seed=i)
        if r["rms_px"] < max(8, 2.5 * max(sigmas)):      # drop fits that did not converge
            out.append((Wj, r["H"], r["D"], Wj * r["H"] * r["D"]))
    return np.array(out)
