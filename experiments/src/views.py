"""Image views used throughout the study: full frame, leaf-only, background-only and rectified leaf tiles."""
import cv2
import numpy as np

GRAY = np.array([124, 116, 104], np.uint8)  # ImageNet mean colour; maps to ~0 after normalisation


def full_view(rgb):
    return rgb


def leaf_view(rgb, m):
    return np.where(m[..., None], rgb, GRAY).astype(np.uint8)


def bg_view(rgb, m):
    # dilate so the blade silhouette edge (and its width) leaks as little as possible
    md = cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    return np.where(md[..., None], GRAY, rgb).astype(np.uint8)


def bg_patch_view(rgb, m, cell=128):
    """2x2 mosaic of grid cells that contain no blade pixels: board texture without any leaf silhouette."""
    md = cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8))
    H, W = m.shape
    cells = [(y, x) for y in range(0, H - cell + 1, cell) for x in range(0, W - cell + 1, cell)
             if md[y:y + cell, x:x + cell].sum() == 0]
    if not cells:
        return np.tile(GRAY, (2 * cell, 2 * cell, 1))
    idx = np.linspace(0, len(cells) - 1, 4).round().astype(int)
    p = [rgb[y:y + cell, x:x + cell] for y, x in (cells[i] for i in idx)]
    return np.vstack([np.hstack(p[:2]), np.hstack(p[2:])])


def rectify(rgb, m):
    """Rotate so the blade's long axis is horizontal and crop to its bounding band.

    Returns the rectified RGB band (background greyed) and its mask.
    """
    ys, xs = np.nonzero(m)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    mean = pts.mean(0)
    _, vecs = np.linalg.eigh(np.cov((pts - mean).T))
    major = vecs[:, 1]
    ang = np.degrees(np.arctan2(major[1], major[0]))
    H, W = m.shape
    R = cv2.getRotationMatrix2D((float(mean[0]), float(mean[1])), ang, 1.0)
    # enlarge the canvas so nothing is clipped by the rotation
    c, s = abs(R[0, 0]), abs(R[0, 1])
    nW, nH = int(H * s + W * c), int(H * c + W * s)
    R[0, 2] += nW / 2 - mean[0]
    R[1, 2] += nH / 2 - mean[1]
    rr = cv2.warpAffine(leaf_view(rgb, m), R, (nW, nH), flags=cv2.INTER_LINEAR, borderValue=GRAY.tolist())
    mm = cv2.warpAffine(m.astype(np.uint8), R, (nW, nH), flags=cv2.INTER_NEAREST) > 0
    ys, xs = np.nonzero(mm)
    # trim rows that hold only a few blade pixels so the band height reflects the blade width
    rows = mm.sum(1)
    keep = np.nonzero(rows >= 0.3 * rows.max())[0]
    y0, y1 = keep.min(), keep.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return rr[y0:y1, x0:x1], mm[y0:y1, x0:x1]


def leaf_tiles(rgb, m, k=4):
    """k square tiles (side = band height) spaced evenly along the rectified blade."""
    band, bm = rectify(rgb, m)
    h, w = bm.shape
    side = min(h, w)
    starts = np.linspace(0, max(w - side, 0), k).round().astype(int)
    return [band[:side, x:x + side] for x in starts]
