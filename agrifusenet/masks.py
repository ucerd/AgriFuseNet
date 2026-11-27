"""Colour-based organ segmentation and mask storage.

The reference boards are achromatic (white, black, grey) or navy, whereas the organs are green, yellow or
red-brown. A hue/saturation/value gate followed by morphology gives a seed mask that GrabCut extends to
desaturated organ regions; the whitish midrib of the leaf blade is restored by closing and hole filling.
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def leaf_mask(rgb):
    """Binary mask of the leaf blade in an RGB uint8 image."""
    h, s, v = _hsv(rgb)
    m = ((h >= 12) & (h <= 88) & (s >= 0.10) & (v >= 0.20)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _disk(5))
    m = _keep_large(m, 0.01)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _disk(31))
    m = _fill_holes(m)
    m = _keep_large(m, 0.03)
    return _grabcut_refine(rgb, m, h, s, v)


def organ_mask(rgb, organ):
    """Binary mask of the organ in the other categories. Stalk pieces (bud, cross-section) can be green, yellow,
    orange or red-brown, so every hue except that of the navy board is accepted; leaf tops partly lie on soil,
    so their hue floor excludes brown."""
    if organ == "leavePattern":
        lo, hi, smin, close, hole = 20, 88, 0.12, 5, 0.001
    else:
        lo, hi, smin, close, hole = 0, 94, 0.14, 15, 0.05
    h, s, v = _hsv(rgb)
    hue_ok = ((h >= lo) & (h <= hi)) | ((h >= 165) if organ != "leavePattern" else False)
    m = (hue_ok & (s >= smin) & (v >= 0.20)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _disk(5))
    m = _keep_large(m, 0.005)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _disk(close))
    m = _fill_holes(m, hole)
    m = _keep_large(m, 0.005)
    return _grabcut_refine(rgb, m, h, s, v, close=max(close, 5), hole=hole)


def colour_mask(rgb, organ):
    return leaf_mask(rgb) if organ == "leafwidth" else organ_mask(rgb, organ)


# ----------------------------------------------------------------------------------------------- storage
def mask_file(root, relative_path):
    """Masks mirror the dataset tree: <root>/<category>/<variety>/<image name>.png"""
    return Path(root) / Path(relative_path).with_suffix(".png")


def save_mask(root, relative_path, m):
    f = mask_file(root, relative_path)
    f.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(m.astype(np.uint8) * 255).save(f)


def load_mask(root, relative_path):
    return np.asarray(Image.open(mask_file(root, relative_path))) > 0


# ----------------------------------------------------------------------------------------------- helpers
def _hsv(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)  # OpenCV hue scale 0-180
    return hsv[..., 0].astype(int), hsv[..., 1] / 255.0, hsv[..., 2] / 255.0


def _disk(d):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def _grabcut_refine(rgb, m, h, s, v, size=256, close=21, hole=0.02):
    """Extend the colour seed with GrabCut; achromatic, dark and navy pixels are definite background."""
    board = (s < 0.035) | (v < 0.15) | ((h >= 95) & (h <= 140) & (s >= 0.15))
    lab = np.full(m.shape, cv2.GC_PR_BGD, np.uint8)
    lab[board] = cv2.GC_BGD
    lab[m > 0] = cv2.GC_PR_FGD
    lab[cv2.erode(m, np.ones((9, 9), np.uint8)) > 0] = cv2.GC_FGD
    H, W = m.shape
    small = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    lab_s = cv2.resize(lab, (size, size), interpolation=cv2.INTER_NEAREST)
    if not (lab_s == cv2.GC_FGD).any():
        return m
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), lab_s, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    r = np.isin(lab_s, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    r = cv2.resize(r, (W, H), interpolation=cv2.INTER_NEAREST)
    r = cv2.morphologyEx(r, cv2.MORPH_OPEN, _disk(7))
    r = cv2.morphologyEx(r, cv2.MORPH_CLOSE, _disk(close))
    return _keep_large(_fill_holes(r, hole), 0.03)


def _keep_large(m, frac):
    """Keep connected components larger than frac of the image and at least a quarter of the largest one."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    big = stats[1:, cv2.CC_STAT_AREA].max() if n > 1 else 0
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= max(frac * m.size, 0.25 * big)]
    if not keep and n > 1:
        keep = [1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))]
    return np.isin(lab, keep).astype(np.uint8)


def _fill_holes(m, max_frac=0.02):
    """Fill background components that do not touch the image border and are smaller than max_frac."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(1 - m, 4)
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    holes = [i for i in range(1, n) if i not in border and stats[i, cv2.CC_STAT_AREA] <= max_frac * m.size]
    return m | np.isin(lab, holes).astype(np.uint8)
