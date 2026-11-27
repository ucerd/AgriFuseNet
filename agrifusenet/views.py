"""Image views: full frame, masked organ, background only, background cells, blade tiles, corners, organ crop."""
import cv2
import numpy as np

GRAY = np.array([124, 116, 104], np.uint8)  # ImageNet mean colour; close to zero after normalisation


def masked_view(rgb, m):
    """Organ pixels only; everything else neutral grey."""
    return np.where(m[..., None], rgb, GRAY).astype(np.uint8)


def background_view(rgb, m, dilate=15):
    """Frame with the (dilated) organ greyed out, so that the organ silhouette is hidden as far as possible."""
    md = cv2.dilate(m.astype(np.uint8), np.ones((dilate, dilate), np.uint8)) > 0
    return np.where(md[..., None], GRAY, rgb).astype(np.uint8)


def background_cells(rgb, m, cell=128):
    """2 x 2 mosaic of grid cells that contain no organ pixel: board texture without any organ silhouette."""
    md = cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8))
    H, W = m.shape
    cells = [(y, x) for y in range(0, H - cell + 1, cell) for x in range(0, W - cell + 1, cell)
             if md[y:y + cell, x:x + cell].sum() == 0]
    if not cells:
        return np.tile(GRAY, (2 * cell, 2 * cell, 1))
    idx = np.linspace(0, len(cells) - 1, 4).round().astype(int)
    p = [rgb[y:y + cell, x:x + cell] for y, x in (cells[i] for i in idx)]
    return np.vstack([np.hstack(p[:2]), np.hstack(p[2:])])


def corner_mosaic(rgb, c=112):
    """2 x 2 mosaic of the four corner patches of the frame (a mask-free background probe)."""
    H, W = rgb.shape[:2]
    p = [rgb[:c, :c], rgb[:c, W - c:], rgb[H - c:, :c], rgb[H - c:, W - c:]]
    return np.vstack([np.hstack(p[:2]), np.hstack(p[2:])])


def rectify(rgb, m):
    """Rotate the blade so that its long axis is horizontal and crop it to its band.

    Returns the rectified RGB band (background grey) and its mask.
    """
    ys, xs = np.nonzero(m)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    mean = pts.mean(0)
    _, vecs = np.linalg.eigh(np.cov((pts - mean).T))
    major = vecs[:, 1]
    ang = np.degrees(np.arctan2(major[1], major[0]))
    H, W = m.shape
    R = cv2.getRotationMatrix2D((float(mean[0]), float(mean[1])), ang, 1.0)
    c, s = abs(R[0, 0]), abs(R[0, 1])
    nW, nH = int(H * s + W * c), int(H * c + W * s)  # canvas large enough for the rotated frame
    R[0, 2] += nW / 2 - mean[0]
    R[1, 2] += nH / 2 - mean[1]
    rr = cv2.warpAffine(masked_view(rgb, m), R, (nW, nH), flags=cv2.INTER_LINEAR, borderValue=GRAY.tolist())
    mm = cv2.warpAffine(m.astype(np.uint8), R, (nW, nH), flags=cv2.INTER_NEAREST) > 0
    ys, xs = np.nonzero(mm)
    rows = mm.sum(1)
    keep = np.nonzero(rows >= 0.3 * rows.max())[0]  # band height follows the blade width
    y0, y1 = keep.min(), keep.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return rr[y0:y1, x0:x1], mm[y0:y1, x0:x1]


def blade_tiles(rgb, m, k=4):
    """k square tiles (side = band height) spaced evenly along the rectified blade; they resolve the venation."""
    band, bm = rectify(rgb, m)
    h, w = bm.shape
    side = min(h, w)
    starts = np.linspace(0, max(w - side, 0), k).round().astype(int)
    return [band[:side, x:x + side] for x in starts]


def organ_crop(rgb, m, margin=0.10):
    """Square crop centred on the organ mask with a relative margin; pixels outside the mask are grey."""
    if not m.any():
        return np.ascontiguousarray(masked_view(rgb, m))
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(max(y1 - y0, x1 - x0) * (1 + 2 * margin))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    H, W = m.shape
    canvas = np.tile(GRAY, (side, side, 1)).astype(np.uint8)
    fg = masked_view(rgb, m)
    sy0, sx0 = cy - side // 2, cx - side // 2
    ya, xa = max(sy0, 0), max(sx0, 0)
    yb, xb = min(sy0 + side, H), min(sx0 + side, W)
    canvas[ya - sy0:yb - sy0, xa - sx0:xb - sx0] = fg[ya:yb, xa:xb]
    return canvas
