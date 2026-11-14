"""Pictures of the four steps of the automatic corner finder, for the UI and the README."""
import cv2
import numpy as np

import autocorners as ac

CAPTIONS = ["1  Silhouette: frame minus empty background",
            "2  Front face: crate blue inside the silhouette, one line per side",
            "3  Side bottom edge: top of the sunlit pallet wood, far corner where it stops",
            "4  Side top edge: follows the crate layer joints to one vanishing point"]


def _cap(img, text):
    cv2.rectangle(img, (0, 985), (1920, 1080), (0, 0, 0), -1)
    cv2.putText(img, text, (40, 1048), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
    return img


def _tint(img, mask, colour):
    out = (img * 0.45).astype(np.uint8)
    m = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    out[m] = (0.6 * img[m] + 0.4 * np.array(colour)).astype(np.uint8)
    return out


def _dot(img, p, label):
    c = (int(p[0]), int(p[1]))
    cv2.circle(img, c, 13, (0, 0, 0), -1); cv2.circle(img, c, 10, (0, 220, 255), -1)
    cv2.putText(img, label, (c[0] + 16, c[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)


def _line(img, p, q, colour=(0, 255, 255), extend=0.0):
    d = q - p
    cv2.line(img, tuple((p - extend * d).astype(int)), tuple((q + extend * d).astype(int)), colour, 3, cv2.LINE_AA)


def step_images(uri, idx, corners, layers=5):
    """Four annotated frames showing what each step saw. corners = the automatic corners of frame idx."""
    bg, frames = ac.load(uri, {idx})
    f = frames[idx]; sm = ac.small(f)
    P = {k: np.array(corners[k], float) for k in "ABCDEF"}
    sil = ac.silhouette(sm, bg)
    one = _cap(_tint(f, sil > 0, (255, 255, 255)), CAPTIONS[0])

    two = _tint(f, ac.front_mask(sm, sil), (0, 200, 0))
    for a, b in ("AB", "BC", "CD", "DA"):
        _line(two, P[a], P[b], extend=0.12)
    for k in "ABCD":
        _dot(two, P[k], k)
    _cap(two, CAPTIONS[1])

    h, s, v = cv2.split(cv2.cvtColor(f, cv2.COLOR_BGR2HSV))
    wood = (h >= ac.WOOD_H[0]) & (h <= ac.WOOD_H[1]) & (s >= ac.WOOD_S_MIN) & (v >= ac.WOOD_V_MIN)
    roi = np.zeros_like(wood)
    roi[int(min(P["A"][1], P["E"][1])) - 12:int(max(P["A"][1], P["E"][1])) + 70, int(P["E"][0]) - 10:int(P["A"][0]) + 10] = True
    three = _tint(f, wood & roi, (0, 140, 255))
    _line(three, P["E"], P["A"], extend=0.1)
    _dot(three, P["A"], "A"); _dot(three, P["E"], "E")
    _cap(three, CAPTIONS[2])

    four = (f * 0.6).astype(np.uint8)
    for k in range(layers + 1):
        _line(four, P["A"] + (P["D"] - P["A"]) * k / layers, P["E"] + (P["F"] - P["E"]) * k / layers,
              (0, 255, 255) if k in (0, layers) else (0, 220, 0))
    _line(four, P["E"], P["F"])
    for k in "ADEF":
        _dot(four, P[k], k)
    _cap(four, CAPTIONS[3])
    return [one, two, three, four]
