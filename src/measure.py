"""Turn marked corners into a volume, with a range and a counting cross-check."""
import numpy as np
from boxfit import fit_joint, monte_carlo_joint, ORDER

PALLET_L, PALLET_W, PALLET_H = 1.2, 0.8, 0.144   # EUR pallet, metres
PALLETS_ACROSS = 2                                # counted on the front face
LAYERS = 5                                        # counted on the front face
W_LOAD = PALLETS_ACROSS * PALLET_L                # known width = our ruler


def measure(ann, size, n_mc=200):
    """ann: {"240": {"A": [x, y], ..., "sigma_px": 4}, ...}"""
    used = sorted(int(k) for k in ann if str(k).isdigit())
    corners = [{p: tuple(ann[str(i)][p]) for p in ORDER} for i in used]
    sigmas = [ann[str(i)].get("sigma_px", 5) for i in used]
    fit = fit_joint(corners, W_LOAD, size)
    H, D = fit["H"], fit["D"]
    pallets_deep = max(1, int(round(D / PALLET_W)))
    foot = W_LOAD * pallets_deep * PALLET_W
    res = dict(
        measured_frames=used,
        box_fit=dict(width_m=W_LOAD, depth_m=round(D, 2), height_m=round(H, 2),
                     volume_m3=round(W_LOAD * D * H, 1), focal_px=round(fit["f"]),
                     reproj_rms_px=round(fit["rms_px"], 1)),
        counting_check=dict(pallets_across=PALLETS_ACROSS, pallets_deep=pallets_deep, layers=LAYERS,
                            footprint_m2=round(foot, 2), crate_layer_height_m=round(H / LAYERS, 3),
                            volume_m3=round(foot * H, 2), volume_incl_pallets_m3=round(foot * (H + PALLET_H), 2)),
        reproj={str(i): r for i, r in zip(used, fit["reproj"])},
    )
    if n_mc:
        mc = monte_carlo_joint(corners, sigmas, W_LOAD, size, n=n_mc)
        if len(mc) >= 10:
            v = np.percentile(mc[:, 3], [5, 50, 95])
            res["uncertainty"] = dict(
                volume_m3_p05=round(v[0], 1), volume_m3_p50=round(v[1], 1), volume_m3_p95=round(v[2], 1),
                height_m_p05_p95=[round(x, 2) for x in np.percentile(mc[:, 1], [5, 95])],
                depth_m_p05_p95=[round(x, 2) for x in np.percentile(mc[:, 2], [5, 95])], runs=len(mc))
    return res
