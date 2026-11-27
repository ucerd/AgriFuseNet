"""Botanical descriptors of the leaf blade (164-D), computed on blade pixels of the rectified blade only.

Groups (column prefixes):
  col_   colour moments in RGB, HSV and CIE L*a*b*, circular hue statistics and chromatic vegetation indices
  glcm_  GLCM statistics along (0 deg), across (90 deg) and diagonal to the venation at three distances
  lbp_   uniform local binary pattern histograms at (8, 1) and (16, 2)
  gab_   Gabor energy along, across and diagonal to the venation at three wavelengths
  ven_   band energies of the spectrum of the across-blade intensity profile (venation)
  mor_   scale-invariant morphometrics (taper, width variation, aspect, midrib ratio, edge roughness)

baseline_descriptor() is a conventional 34-D whole-image descriptor (GLCM, LBP, colour) used for comparison.
"""
import cv2
import numpy as np
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from .views import rectify

GROUPS = ("col", "glcm", "lbp", "gab", "ven", "mor")
GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")


def botanical_descriptors(rgb, m):
    """dict of 164 descriptors of the blade in image rgb with mask m."""
    band, bm = rectify(rgb, m)
    gray = cv2.cvtColor(band, cv2.COLOR_RGB2GRAY)
    f = {}
    f.update(colour_features(band, bm))
    f.update(glcm_features(gray, bm))
    f.update(lbp_features(gray, bm))
    f.update(gabor_features(gray, bm))
    f.update(venation_features(gray, bm))
    f.update(morphometric_features(gray, bm))
    return f


def colour_features(rgb, m):
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
    f["col_H_R"] = np.hypot(f["col_H_cos"], f["col_H_sin"])  # circular concentration of hue
    for name, idx in [("NGI", gn), ("ExG", 2 * gn - rn - bn), ("ExR", 1.4 * rn - gn),
                      ("VARI", (g - r) / (g + r - b + 1e-6)), ("GLI", (2 * g - r - b) / (2 * g + r + b + 1e-6))]:
        idx = np.clip(idx, -5, 5)
        f[f"col_{name}_mean"], f[f"col_{name}_std"] = idx.mean(), idx.std()
        for q in (10, 50, 90):
            f[f"col_{name}_p{q}"] = np.percentile(idx, q)
    f["col_ExGR_mean"] = f["col_ExG_mean"] - f["col_ExR_mean"]
    return f


def glcm_features(gray, m):
    q = (gray // 8).astype(np.uint8) + 1  # 32 grey levels; level 0 marks background
    q[~m] = 0
    P = graycomatrix(q, distances=[1, 3, 6], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=33)
    P = P[1:, 1:].astype(np.float64)  # co-occurrences with background are discarded
    P /= P.sum(axis=(0, 1), keepdims=True) + 1e-12
    f = {}
    for prop in GLCM_PROPS:
        v = graycoprops(P, prop)  # (distances, angles)
        for di, d in enumerate((1, 3, 6)):
            f[f"glcm_{prop}_d{d}_along"] = v[di, 0]
            f[f"glcm_{prop}_d{d}_across"] = v[di, 2]
            f[f"glcm_{prop}_d{d}_diag"] = (v[di, 1] + v[di, 3]) / 2
    return f


def lbp_features(gray, m):
    f = {}
    er = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0  # codes do not straddle the blade edge
    for P, R in ((8, 1), (16, 2)):
        code = local_binary_pattern(gray, P, R, "uniform")[er]
        h = np.bincount(code.astype(int), minlength=P + 2)[:P + 2].astype(float)
        h /= h.sum() + 1e-12
        for i, v in enumerate(h):
            f[f"lbp_P{P}R{R}_b{i}"] = v
    return f


def gabor_features(gray, m):
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


def venation_features(gray, m, nb=12):
    """Spectrum of the across-blade intensity profile (blade pixels averaged along the blade length)."""
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


def morphometric_features(band_gray, bm):
    widths = bm.sum(0).astype(float)
    widths = widths[widths > 0.3 * widths.max()]
    W = np.median(widths)
    f = {"mor_taper": (np.percentile(widths, 90) - np.percentile(widths, 10)) / W,
         "mor_width_cv": widths.std() / W,
         "mor_aspect": bm.shape[1] / W}
    # midrib: the brightest ridge of the smoothed across-blade profile in the central 60 % of the band
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
    # edge roughness: high-frequency residual of the upper and lower blade edges
    cols = np.nonzero(bm.any(0))[0]
    top = np.array([np.argmax(bm[:, c]) for c in cols], float)
    bot = np.array([bm.shape[0] - 1 - np.argmax(bm[::-1, c]) for c in cols], float)
    rough = [np.abs(e - cv2.blur(e[None], (31, 1))[0]).mean() / W for e in (top, bot)]
    f["mor_edge_rough"] = float(np.mean(rough))
    return f


def baseline_descriptor(rgb):
    """34-D whole-image descriptor: GLCM (4 properties x 4 angles, distance 1), uniform LBP (P=8, R=1),
    mean and standard deviation of NGI and ExG, and mean R, G, B and V."""
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    P = graycomatrix(gray // 8, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=32, symmetric=True, normed=True)
    f = {f"glcm_{p}_{a}": graycoprops(P, p)[0, a] for p in ("contrast", "energy", "homogeneity", "correlation")
         for a in range(4)}
    h = np.bincount(local_binary_pattern(gray, 8, 1, "uniform").astype(int).ravel(), minlength=10)[:10]
    f.update({f"lbp_{i}": v for i, v in enumerate(h / h.sum())})
    x = rgb.reshape(-1, 3).astype(np.float32)
    s = x.sum(1) + 1e-6
    ngi, exg = x[:, 1] / s, (2 * x[:, 1] - x[:, 0] - x[:, 2]) / s
    f.update({"ngi_m": ngi.mean(), "ngi_s": ngi.std(), "exg_m": exg.mean(), "exg_s": exg.std()})
    f.update({f"mean_{c}": x[:, i].mean() for i, c in enumerate("RGB")})
    f["mean_V"] = x.max(1).mean()
    return f
