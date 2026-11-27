"""Leaf-only, vein-aligned handcrafted descriptors computed on the rectified blade.

Groups (each returned with a name prefix so ablations can select them):
  col_*   colour statistics and vegetation indices over blade pixels
  glcm_*  GLCM statistics along (0 deg) and across (90 deg) the venation, plus diagonals
  lbp_*   uniform LBP histograms over blade pixels at two scales
  gab_*   Gabor energy parallel / perpendicular to the venation at several wavelengths
  ven_*   venation spectrum of the cross-blade intensity profile
  mor_*   scale-invariant morphometrics (midrib/blade width ratio, taper, edge roughness)
"""
import time
import cv2
import numpy as np
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from common import load_manifest, load_rgb, CACHE
from segment import load_mask
from views import rectify

GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")


def colour_feats(rgb, m):
    px = rgb[m].astype(np.float32)
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    s = r + g + b + 1e-6
    rn, gn, bn = r / s, g / s, b / s
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV_FULL)[m].astype(np.float32)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[m].astype(np.float32)
    ang = hsv[:, 0] / 256 * 2 * np.pi
    f = {}
    for name, ch in [("R", r), ("G", g), ("B", b), ("S", hsv[:, 1]), ("V", hsv[:, 2]),
                     ("L", lab[:, 0]), ("a", lab[:, 1]), ("b", lab[:, 2])]:
        f[f"col_{name}_mean"], f[f"col_{name}_std"], f[f"col_{name}_skew"] = ch.mean(), ch.std(), skew(ch)
    f["col_H_cos"], f["col_H_sin"] = np.cos(ang).mean(), np.sin(ang).mean()
    f["col_H_R"] = np.hypot(f["col_H_cos"], f["col_H_sin"])  # circular concentration
    for name, idx in [("NGI", gn), ("ExG", 2 * gn - rn - bn), ("ExR", 1.4 * rn - gn),
                      ("VARI", (g - r) / (g + r - b + 1e-6)), ("GLI", (2 * g - r - b) / (2 * g + r + b + 1e-6))]:
        idx = np.clip(idx, -5, 5)
        f[f"col_{name}_mean"], f[f"col_{name}_std"] = idx.mean(), idx.std()
        for q in (10, 50, 90):
            f[f"col_{name}_p{q}"] = np.percentile(idx, q)
    f["col_ExGR_mean"] = f["col_ExG_mean"] - f["col_ExR_mean"]
    return f


def glcm_feats(gray, m):
    q = (gray // 8).astype(np.uint8) + 1  # 32 grey levels, 0 reserved for background
    q[~m] = 0
    P = graycomatrix(q, distances=[1, 3, 6], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=33)
    P = P[1:, 1:].astype(np.float64)  # drop background co-occurrences
    P /= P.sum(axis=(0, 1), keepdims=True) + 1e-12
    f = {}
    for prop in GLCM_PROPS:
        v = graycoprops(P, prop)  # (distances, angles)
        for di, d in enumerate((1, 3, 6)):
            f[f"glcm_{prop}_d{d}_along"] = v[di, 0]
            f[f"glcm_{prop}_d{d}_across"] = v[di, 2]
            f[f"glcm_{prop}_d{d}_diag"] = (v[di, 1] + v[di, 3]) / 2
    return f


def lbp_feats(gray, m):
    f = {}
    er = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0  # avoid codes straddling the blade edge
    for P, R in ((8, 1), (16, 2)):
        code = local_binary_pattern(gray, P, R, "uniform")[er]
        h = np.bincount(code.astype(int), minlength=P + 2)[:P + 2].astype(float)
        h /= h.sum() + 1e-12
        for i, v in enumerate(h):
            f[f"lbp_P{P}R{R}_b{i}"] = v
    return f


def gabor_feats(gray, m):
    f = {}
    g = gray.astype(np.float32) / 255
    er = cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    for lam in (4, 8, 16):
        for name, theta in (("along", 0.0), ("across", np.pi / 2), ("diag", np.pi / 4)):
            k = cv2.getGaborKernel((31, 31), sigma=0.56 * lam, theta=theta, lambd=lam, gamma=0.5, psi=0)
            k -= k.mean()
            e = cv2.filter2D(g, cv2.CV_32F, k)[er]
            f[f"gab_l{lam}_{name}_mean"], f[f"gab_l{lam}_{name}_std"] = np.abs(e).mean(), e.std()
    return f


def venation_feats(gray, m, nb=12):
    """Spectrum of the across-blade intensity profile (columns averaged along the blade length)."""
    g = gray.astype(np.float32)
    prof = np.array([g[y][m[y]].mean() if m[y].any() else np.nan for y in range(g.shape[0])])
    prof = prof[~np.isnan(prof)]
    prof = cv2.resize(prof[None].astype(np.float32), (128, 1), interpolation=cv2.INTER_LINEAR)[0]
    prof = (prof - prof.mean()) / (prof.std() + 1e-6)
    spec = np.abs(np.fft.rfft(prof))[1:]
    edges = np.geomspace(1, len(spec), nb + 1).astype(int)
    f = {f"ven_band{i}": spec[edges[i]:max(edges[i + 1], edges[i] + 1)].mean() for i in range(nb)}
    f["ven_peak_freq"] = float(np.argmax(spec[2:]) + 3)
    return f


def morph_feats(band_gray, bm):
    widths = bm.sum(0).astype(float)
    widths = widths[widths > 0.3 * widths.max()]
    W = np.median(widths)
    f = {"mor_taper": (np.percentile(widths, 90) - np.percentile(widths, 10)) / W,
         "mor_width_cv": widths.std() / W,
         "mor_aspect": bm.shape[1] / W}
    # midrib = the brightest smooth ridge of the across-blade profile near the blade centre
    g = band_gray.astype(np.float32)
    prof = np.array([g[y][bm[y]].mean() if bm[y].any() else np.nan for y in range(g.shape[0])])
    prof = np.nan_to_num(prof, nan=np.nanmean(prof))
    prof = cv2.GaussianBlur(prof[:, None], (1, 5), 0)[:, 0]
    n = len(prof)
    c0, c1 = int(0.2 * n), int(0.8 * n)
    pk = c0 + int(np.argmax(prof[c0:c1]))
    base = np.median(prof)
    half = base + 0.5 * (prof[pk] - base)
    lo, hi = pk, pk
    while lo > 0 and prof[lo] > half:
        lo -= 1
    while hi < n - 1 and prof[hi] > half:
        hi += 1
    f["mor_midrib_ratio"] = (hi - lo) / n
    f["mor_midrib_contrast"] = (prof[pk] - base) / (base + 1e-6)
    f["mor_midrib_offset"] = abs(pk / n - 0.5)
    # edge roughness: high-frequency residual of the upper/lower blade edges
    cols = np.nonzero(bm.any(0))[0]
    top = np.array([np.argmax(bm[:, c]) for c in cols], float)
    bot = np.array([bm.shape[0] - 1 - np.argmax(bm[::-1, c]) for c in cols], float)
    rough = [np.abs(e - cv2.blur(e[None], (31, 1))[0]).mean() / W for e in (top, bot)]
    f["mor_edge_rough"] = float(np.mean(rough))
    return f


def extract(rgb, m):
    band, bm = rectify(rgb, m)
    gray = cv2.cvtColor(band, cv2.COLOR_RGB2GRAY)
    f = {}
    f.update(colour_feats(band, bm))
    f.update(glcm_feats(gray, bm))
    f.update(lbp_feats(gray, bm))
    f.update(gabor_feats(gray, bm))
    f.update(venation_feats(gray, bm))
    f.update(morph_feats(gray, bm))
    return f


def legacy_extract(rgb, m=None):
    """Re-implementation of the original 34-D descriptor (whole image, no mask), for comparison.

    GLCM: 4 props x 4 angles at distance 1 (16) + uniform LBP P=8,R=1 (10, the 26-bin figure used P=24)
    + colour: mean/std of NGI and ExG (4) + mean of R,G,B,V (4)  = 34.
    """
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    P = graycomatrix(gray // 8, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=32, symmetric=True, normed=True)
    f = {f"glcm_{p}_{a}": graycoprops(P, p)[0, a] for p in ("contrast", "energy", "homogeneity", "correlation") for a in range(4)}
    h = np.bincount(local_binary_pattern(gray, 8, 1, "uniform").astype(int).ravel(), minlength=10)[:10]
    f.update({f"lbp_{i}": v for i, v in enumerate(h / h.sum())})
    x = rgb.reshape(-1, 3).astype(np.float32)
    s = x.sum(1) + 1e-6
    ngi, exg = x[:, 1] / s, (2 * x[:, 1] - x[:, 0] - x[:, 2]) / s
    f.update({"ngi_m": ngi.mean(), "ngi_s": ngi.std(), "exg_m": exg.mean(), "exg_s": exg.std()})
    f.update({f"mean_{c}": x[:, i].mean() for i, c in enumerate("RGB")})
    f["mean_V"] = x.max(1).mean()
    return f


if __name__ == "__main__":
    import pandas as pd
    df = load_manifest("leafwidth")
    rows, legacy, t0 = [], [], time.time()
    for i, (rel, p) in enumerate(zip(df.relative_path, df.path)):
        a, m = load_rgb(p), load_mask(rel)
        rows.append(extract(a, m)); legacy.append(legacy_extract(a))
        if i % 100 == 0:
            print(i, "%.1fs" % (time.time() - t0), flush=True)
    H = pd.DataFrame(rows); L = pd.DataFrame(legacy)
    print("handcrafted dims", H.shape, "legacy dims", L.shape, "nan", int(H.isna().sum().sum()))
    H.to_parquet(CACHE / "handcrafted.parquet"); L.to_parquet(CACHE / "handcrafted_legacy.parquet")
    print(H.columns.str.split("_").str[0].value_counts().to_dict())
