"""Colour-based leaf-blade segmentation against the reference boards.

The boards are achromatic (white/black/grey) or navy, while the blade is yellow-green, so a
hue/saturation gate followed by morphology recovers the blade; the whitish midrib is
filled back in by closing and hole filling.
"""
import sys
import cv2
import numpy as np
from PIL import Image
from common import load_manifest, load_rgb, CACHE

MASK_DIR = CACHE / "masks"


def leaf_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)  # H in [0,180)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    m = ((h >= 12) & (h <= 88) & (s >= 0.10) & (v >= 0.20)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    m = _keep_large(m, 0.01)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    m = _fill_holes(m)
    m = _keep_large(m, 0.03)
    return _grabcut_refine(rgb, m, h, s, v)


def organ_mask(rgb, organ):
    """Object mask for the other organ categories. Cane pieces (bud, cross-section) can be green, yellow,
    orange or red-brown, so every hue except the navy board passes; leaf tops (leavePattern) lie partly on
    brown soil, so the hue floor is raised above soil."""
    if organ == "leavePattern":  # thin, spread blades: light morphology so the sheet between blades stays out
        lo, hi, smin, close, hole = 20, 88, 0.12, 5, 0.001
    else:
        lo, hi, smin, close, hole = 0, 94, 0.14, 15, 0.05
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    hue_ok = ((h >= lo) & (h <= hi)) | ((h >= 165) if organ != "leavePattern" else False)
    m = (hue_ok & (s >= smin) & (v >= 0.20)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    m = _keep_large(m, 0.005)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
    m = _fill_holes(m, hole)
    m = _keep_large(m, 0.005)
    return _grabcut_refine(rgb, m, h, s, v, close=max(close, 5), hole=hole)


def _grabcut_refine(rgb, m, h, s, v, size=256, close=21, hole=0.02):
    """Extend the colour seed to desaturated blade regions using GrabCut colour models."""
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
    r = cv2.morphologyEx(r, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    r = cv2.morphologyEx(r, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
    return _keep_large(_fill_holes(r, hole), 0.03)


def _keep_large(m, frac):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    big = stats[1:, cv2.CC_STAT_AREA].max() if n > 1 else 0
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= max(frac * m.size, 0.25 * big)]
    if not keep and n > 1:
        keep = [1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))]
    return np.isin(lab, keep).astype(np.uint8)


def _fill_holes(m, max_frac=0.02):
    """Fill small background components that do not touch the image border (e.g. the midrib)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(1 - m, 4)
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    holes = [i for i in range(1, n) if i not in border and stats[i, cv2.CC_STAT_AREA] <= max_frac * m.size]
    return m | np.isin(lab, holes).astype(np.uint8)


def load_mask(rel_path):
    return np.asarray(Image.open(MASK_DIR / (rel_path.replace("/", "__") + ".png"))) > 0


if __name__ == "__main__":
    feature = sys.argv[1] if len(sys.argv) > 1 else "leafwidth"
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    df = load_manifest(feature)
    cover = []
    for rel, p in zip(df.relative_path, df.path):
        m = leaf_mask(load_rgb(p))
        Image.fromarray(m * 255).save(MASK_DIR / (rel.replace("/", "__") + ".png"))
        cover.append(m.mean())
    cover = np.array(cover)
    print("leaf coverage: mean %.3f  min %.3f  max %.3f" % (cover.mean(), cover.min(), cover.max()))
    for q in np.argsort(cover)[:8]:
        print("lowest", df.relative_path[q], round(cover[q], 3))
    for q in np.argsort(cover)[-8:]:
        print("highest", df.relative_path[q], round(cover[q], 3))
