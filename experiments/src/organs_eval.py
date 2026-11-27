"""Feasibility of multi-organ fusion: per-category leakage-free accuracy, background (border) probe, and late
fusion over pseudo-specimens (one image per organ category, drawn within a variety from out-of-fold scores).

Excluded: internodeDistance/CPF-251 and /CPF-253 (the two folders hold the same 200 files) and every image whose
exact file also appears under another variety.

Usage: python organs_eval.py [seeds e.g. 0,1,2] [--feat FILE --views v1,v2 --aug KEY --tag TAG]
  default: full-image features (organs_dinov2_s.npz, views full,border)
  organ-only: --feat organs_obj_dinov2_s.npz --views obj,bgonly --aug aug_obj --tag obj
Writes results/organs_feasibility<_tag>.json and cache/organs_oof<_tag>.npz
"""
import argparse
import sys
import json
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from common import DATA, CACHE, RESULTS, CLASSES
from groups import components

ap = argparse.ArgumentParser()
ap.add_argument("seeds", nargs="?", default="0,1,2")
ap.add_argument("--feat", default="organs_dinov2_s.npz")
ap.add_argument("--views", default="full,border")
ap.add_argument("--aug", default="aug_full")
ap.add_argument("--tag", default="")
A = ap.parse_args()
SEEDS = tuple(int(s) for s in A.seeds.split(","))
VIEWS = A.views.split(",")
TAG = f"_{A.tag}" if A.tag else ""
F = np.load(CACHE / A.feat, allow_pickle=True)
meta = pd.read_csv(DATA / "metadata" / "samples.csv").set_index("relative_path").loc[F["relative_path"]].reset_index()
meta["y"] = meta.variety.map(CLASSES.index)
conflict = meta.groupby("hash_sha256").variety.transform("nunique") > 1
copied = (meta.feature_type == "internodeDistance") & meta.variety.isin(["CPF-251", "CPF-253"])
keep = ~(conflict | copied)
print("excluded:", int((~keep).sum()), "images", flush=True)
CATS = sorted(meta.feature_type.unique())
# near-duplicate groups always from full-frame embeddings, so every view is split identically
_full = np.load(CACHE / "organs_dinov2_s.npz", allow_pickle=True)
_row = {p: i for i, p in enumerate(_full["relative_path"])}
FULL = _full["full"][[_row[p] for p in F["relative_path"]]]
svm = lambda: make_pipeline(StandardScaler(), SVC(C=10, gamma="scale"))


def zscores(model, X, n_cls_present, present):
    """Per-image standardised one-vs-rest margins, written into a 5-class array (absent classes = 0 = neutral)."""
    d = model.decision_function(X)
    d = (d - d.mean(1, keepdims=True)) / (d.std(1, keepdims=True) + 1e-9)
    out = np.zeros((len(X), len(CLASSES)), np.float32)
    out[:, present] = d
    return out


def cv(X, Xa, y, groups, splits):
    """Out-of-fold z-scores; Xa (n_aug, n, d) augmented rows are added for training images only."""
    Z = np.zeros((len(y), len(CLASSES)), np.float32)
    for tr, te in splits:
        Xtr, ytr = X[tr], y[tr]
        if Xa is not None:
            Xtr = np.vstack([Xtr, Xa[:, tr].reshape(-1, X.shape[1])]); ytr = np.concatenate([ytr, np.tile(y[tr], Xa.shape[0])])
        m = svm().fit(Xtr, ytr)
        Z[te] = zscores(m, X[te], None, m.classes_)
    return Z


res, oof = {}, {}
for cat in CATS:
    ix = np.nonzero((meta.feature_type == cat).values & keep.values)[0]
    sub = meta.iloc[ix].reset_index(drop=True)
    y = sub.y.values
    X, B = F[VIEWS[0]][ix], F[VIEWS[1]][ix]
    Xa = F[A.aug][:, ix] if A.aug in F else None
    B_aug = None
    E = normalize(FULL[ix])
    g = components(sub, E @ E.T, 0.985, 10)
    num = sub.relative_path.str.extract(r"img_(\d+)")[0].astype(int).values
    third = np.zeros(len(sub), int)
    for c in np.unique(y):
        jj = np.nonzero(y == c)[0]; jj = jj[np.argsort(num[jj])]
        for k, part in enumerate(np.array_split(jj, 3)):
            third[part] = k
    r = {"n": int(len(y)), "classes": [CLASSES[c] for c in np.unique(y)], "n_groups": int(len(np.unique(g)))}
    for view, XX, XA in ((VIEWS[0], X, Xa), (VIEWS[1], B, B_aug)):
        accs, bals = [], []
        for s in SEEDS:
            sp = list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(XX, y, g))
            Z = cv(XX, XA, y, g, sp)
            p = Z.argmax(1)
            accs.append(accuracy_score(y, p)); bals.append(balanced_accuracy_score(y, p))
            oof[f"{cat}|{view}|grouped|{s}"] = Z
        Zs = cv(XX, XA, y, g, [(np.nonzero(third != k)[0], np.nonzero(third == k)[0]) for k in range(3)])
        oof[f"{cat}|{view}|session"] = Zs
        r[view] = {"grouped_acc": float(np.mean(accs)), "grouped_bal": float(np.mean(bals)),
                   "session_acc": float(accuracy_score(y, Zs.argmax(1))), "session_bal": float(balanced_accuracy_score(y, Zs.argmax(1)))}
    oof[f"{cat}|y"] = y
    res[cat] = r
    v0, v1 = VIEWS
    print(f"{cat:24s} n={r['n']:4d} groups={r['n_groups']:4d} | {v0} grouped {100*r[v0]['grouped_bal']:5.1f} session "
          f"{100*r[v0]['session_bal']:5.1f} | {v1} grouped {100*r[v1]['grouped_bal']:5.1f} session "
          f"{100*r[v1]['session_bal']:5.1f}  (balanced acc.)", flush=True)

# leaf-width: the background-free leaf model of the paper (out-of-fold scores of the reference run)
from common import load_manifest
lw = np.load(RESULTS / "runs" / "abl_dinov2_s_reference_oof.npz")
ylw = load_manifest().y.values
z = lambda a: (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-9)
for s in SEEDS:
    for v in VIEWS:
        oof[f"leaf_model|{v}|grouped|{s}"] = z(lw[f"score_s{s}"])
oof["leaf_model|y"] = ylw
# session-thirds out-of-fold scores of the paper's leaf model (fusion network + SVM), cached
_ls = CACHE / "leaf_model_session_oof.npy"
if not _ls.exists():
    from fusion import FusionSVM
    D = dict(np.load(CACHE / "deep_dinov2_s.npz")); H = pd.read_parquet(CACHE / "handcrafted.parquet").values
    XL = np.hstack([D["leaf"], D["tiles"], H]).astype(np.float32)
    XLa = np.concatenate([D["aug_leaf"], D["aug_tiles"], np.repeat(H[None], 4, 0)], 2).astype(np.float32)
    s0, s1 = D["leaf"].shape[1], 2 * D["leaf"].shape[1]
    sl = [slice(0, s0), slice(s0, s1), slice(s1, XL.shape[1])]
    lm = load_manifest(); numl = lm.relative_path.str.extract(r"img_(\d+)")[0].astype(int).values
    tl = np.zeros(len(lm), int)
    for c in np.unique(ylw):
        jj = np.nonzero(ylw == c)[0]; jj = jj[np.argsort(numl[jj])]
        for k, part in enumerate(np.array_split(jj, 3)):
            tl[part] = k
    ZL = np.zeros((len(ylw), len(CLASSES)), np.float32)
    for k in range(3):
        tr, te = tl != k, tl == k
        Xtr = np.vstack([XL[tr], XLa[:, tr].reshape(-1, XL.shape[1])]); ytr = np.concatenate([ylw[tr], np.tile(ylw[tr], 4)])
        ZL[te] = z(FusionSVM(slices=sl).fit(Xtr, ytr).decision_function(XL[te]))
    np.save(_ls, ZL)
    print("leaf model session acc %.1f" % (100 * (ZL.argmax(1) == ylw).mean()), flush=True)
for v in VIEWS:
    oof[f"leaf_model|{v}|session"] = np.load(_ls)
CATS2 = [c for c in CATS if c != "leafwidth"] + ["leaf_model"]


def specimens(keys, proto, view=None, n_per_class=400, rng_seed=0):
    """Balanced accuracy of pseudo-specimens: per true variety, one random image from each category in `keys`
    that contains the variety; fused score = sum of the categories' standardised margins."""
    view = view or VIEWS[0]
    rng = np.random.default_rng(rng_seed)
    accs = []
    for s in (SEEDS if proto == "grouped" else [None]):
        per_class = []
        for c in range(len(CLASSES)):
            tot = np.zeros((n_per_class, len(CLASSES)))
            used = 0
            for k in keys:
                key = f"{k}|{view}|grouped|{s}" if proto == "grouped" else f"{k}|{view}|session"
                if key not in oof:
                    continue
                yy = oof[f"{k}|y"]
                jj = np.nonzero(yy == c)[0]
                if len(jj) == 0:
                    continue
                tot += oof[key][rng.choice(jj, n_per_class)]
                used += 1
            if used:
                per_class.append((tot.argmax(1) == c).mean())
        accs.append(np.mean(per_class))
    return float(np.mean(accs))


fus = {}
CATS2 = [c for c in CATS if c != "leafwidth"] + ["leaf_model"]
sets = {"leaf model only": ["leaf_model"], "leaf width only": ["leafwidth"],
        "leaf model + leaf pattern": ["leaf_model", "leavePattern"],
        "leaf width + leaf pattern": ["leafwidth", "leavePattern"],
        "all organs (leaf width view)": CATS, "all organs (leaf model)": CATS2}
for name, keys in sets.items():
    for view in VIEWS:
        g, se = specimens(keys, "grouped", view), specimens(keys, "session", view)
        fus[f"{name} [{view}]"] = {"grouped": g, "session": se}
        print(f"FUSION {name:32s} [{view:7s}] grouped {100*g:5.1f} session {100*se:5.1f}", flush=True)
combos = sorted(((specimens(list(c), "session"), c) for r in (2, 3) for c in itertools.combinations(CATS, r)), reverse=True)
for a_, c in combos:
    print(f"COMBO session [{VIEWS[0]}] {100*a_:5.1f} {'+'.join(c)}")
(RESULTS / f"organs_feasibility{TAG}.json").write_text(json.dumps({"per_category": res, "fusion": fus,
                                                                   "combos_session": [[a_, list(c)] for a_, c in combos]}, indent=1))
np.savez_compressed(CACHE / f"organs_oof{TAG}.npz", **{k: v for k, v in oof.items()})
