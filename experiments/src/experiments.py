"""All revision experiments. Each named configuration is evaluated once and cached in results/runs/<name>.json.

python experiments.py <experiment-set> [--seeds 5]
Sets: leakage, audit, compare, ablation, classifiers, backbones
"""
import os
import re
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from common import load_manifest, CACHE, RESULTS
from evalkit import repeated_group_splits, random_splits, run_cv, summarise
from fusion import FusionSVM

warnings.filterwarnings("ignore")
RUNS = RESULTS / "runs"
RUNS.mkdir(exist_ok=True)
df = load_manifest("leafwidth")
y = df.y.values
G = pd.read_csv(CACHE / "groups.csv")
assert (G.relative_path == df.relative_path).all()
groups = G.group.values
SEEDS = (0, 1, 2, 3, 4)
SVM_GRID = {"svc__C": [1, 10, 100], "svc__kernel": ["rbf", "linear"]}

_deep = {}


def deep(backbone, view):
    if backbone not in _deep:
        _deep[backbone] = dict(np.load(CACHE / f"deep_{backbone}.npz"))
    return _deep[backbone][view]


HAND = pd.read_parquet(CACHE / "handcrafted.parquet")
LEGACY = pd.read_parquet(CACHE / "handcrafted_legacy.parquet").values


def hand(groups_=None):
    if groups_ is None:
        return HAND.values
    return HAND.loc[:, HAND.columns.str.split("_").str[0].isin(groups_)].values


def svm_pipe(pca=None, lda=False):
    steps = [StandardScaler()]
    if pca:
        steps.append(PCA(pca, random_state=0))
    if lda:
        steps.append(LinearDiscriminantAnalysis())
    steps.append(SVC(gamma="scale"))
    return make_pipeline(*steps)


def stack(*blocks):
    """Concatenate feature blocks; returns matrix and the column slices of each block."""
    sl, s = [], 0
    for b in blocks:
        sl.append(slice(s, s + b.shape[1])); s += b.shape[1]
    return np.hstack(blocks).astype(np.float32), sl


def aug_stack(*aug_blocks):
    """Stack (n_aug, N, D) blocks; static blocks (N, D) are repeated for each augmentation."""
    n_aug = max(b.shape[0] for b in aug_blocks if b.ndim == 3)
    return np.concatenate([b if b.ndim == 3 else np.repeat(b[None], n_aug, 0) for b in aug_blocks], 2).astype(np.float32)


def selected(name):
    """ONLY=<regex> restricts a run to matching configuration names (one SLURM task per configuration)."""
    only = os.environ.get("ONLY")
    return only is None or re.fullmatch(only, name) is not None


def evaluate(name, make_model, X, X_aug=None, grid=None, protocol="group", seeds=SEEDS, extra=None, force=False):
    path = RUNS / f"{name}.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    if not selected(name):
        return None
    if protocol == "group":
        splits = repeated_group_splits(y, groups, seeds=seeds)
    elif protocol == "groupdup":  # sensitivity: group only exact / near-identical duplicates
        splits = repeated_group_splits(y, G.group_dup.values, seeds=seeds)
    else:
        splits = random_splits(y, seeds=seeds)
    t0 = time.time()
    res = run_cv(make_model, X, y, splits, groups=groups, X_aug=X_aug, param_grid=grid)
    out = {"name": name, "protocol": protocol, "dim": None if callable(X) else int(X.shape[1]),
           "summary": summarise(res, y), "folds": res["folds"], "wall_s": time.time() - t0, **(extra or {})}
    np.savez_compressed(RUNS / f"{name}_oof.npz", **{f"score_s{s}": o["score"] for s, o in res["oof"].items()},
                        **{f"pred_s{s}": o["pred"] for s, o in res["oof"].items()})
    path.write_text(json.dumps(out, default=float))
    s = out["summary"]
    print(f"{name:45s} acc {100*s['acc']['mean']:6.2f}±{100*s['acc']['std']:.2f}  F1 {100*s['macro_f1']['mean']:6.2f}"
          f"  AUC {s['macro_auc']['mean']:.4f}  ({out['wall_s']:.0f}s)", flush=True)
    return out


# ----------------------------------------------------------------------------------------------- sets

def run_leakage():
    """Same representations under the leaky random protocol and the grouped protocol."""
    for proto in ("random", "group"):
        evaluate(f"leak_{proto}_resnet50_full", svm_pipe, deep("resnet50", "full"), grid=SVM_GRID, protocol=proto)
        evaluate(f"leak_{proto}_legacy34", svm_pipe, LEGACY, grid=SVM_GRID, protocol=proto)
        evaluate(f"leak_{proto}_hand164", svm_pipe, hand(), grid=SVM_GRID, protocol=proto)


def run_audit(*backbones):
    """Background-shortcut audit: what each view alone supports, per backbone."""
    for b in backbones or ("resnet50", "dinov2_s", "effnet_b0", "convnext_t", "mobilenetv3"):
        for v in ("full", "bg", "bgpatch", "leaf", "tiles"):
            evaluate(f"audit_{b}_{v}", svm_pipe, deep(b, v), grid=SVM_GRID)
    evaluate("audit_legacy34_fullimage", svm_pipe, LEGACY, grid=SVM_GRID)
    evaluate("audit_hand164_leafonly", svm_pipe, hand(), grid=SVM_GRID)


def fusion_model(slices, **kw):
    return lambda: FusionSVM(slices=slices, **kw)


def proposed_inputs(b, hand_groups=None, use_aug=True, branches=("leaf", "tiles", "hand")):
    blocks, aug_blocks = [], []
    for br in branches:
        if br == "hand":
            h = hand(hand_groups); blocks.append(h); aug_blocks.append(h)
        else:
            blocks.append(deep(b, br)); aug_blocks.append(deep(b, "aug_" + br))
    X, sl = stack(*blocks)
    return X, sl, (aug_stack(*aug_blocks) if use_aug else None)


def run_compare(b="resnet50"):
    """Supervisor's requested comparisons on one backbone, all under the grouped protocol."""
    F, H = deep(b, "full"), hand()
    evaluate(f"cmp_{b}_deep_full", svm_pipe, F, grid=SVM_GRID)
    evaluate(f"cmp_{b}_deep_full_pca128", lambda: svm_pipe(pca=128), F, grid=SVM_GRID)
    evaluate(f"cmp_{b}_deep_full_lda4", lambda: svm_pipe(lda=True), F, grid=SVM_GRID)
    evaluate(f"cmp_legacy34", svm_pipe, LEGACY, grid=SVM_GRID)
    X, sl = stack(F, LEGACY)
    evaluate(f"cmp_{b}_concat_full_legacy", svm_pipe, X, grid=SVM_GRID)
    evaluate(f"cmp_{b}_concat_full_legacy_pca128", lambda: svm_pipe(pca=128), X, grid=SVM_GRID)
    evaluate(f"cmp_{b}_orig_scalar_fusion", fusion_model(sl, mode="scalar", p_drop=0.0, label_smoothing=0.0), X)
    # leaf-based representations
    evaluate(f"cmp_{b}_deep_leaf", svm_pipe, deep(b, "leaf"), grid=SVM_GRID)
    X, sl, Xa = proposed_inputs(b)
    evaluate(f"cmp_{b}_concat_leaf_tiles_hand", svm_pipe, X, grid=SVM_GRID)
    evaluate(f"cmp_{b}_concat_leaf_tiles_hand_pca128", lambda: svm_pipe(pca=128), X, grid=SVM_GRID)
    evaluate(f"cmp_{b}_proposed", fusion_model(sl), X, X_aug=Xa)


def run_ablation(b="resnet50"):
    X, sl, Xa = proposed_inputs(b)
    # the full model re-run on the same hardware as the variants (GPU), as the reference of the ablation table
    evaluate(f"abl_{b}_reference", fusion_model(sl), X, X_aug=Xa)
    evaluate(f"abl_{b}_proposed_noaug", fusion_model(sl), X)
    evaluate(f"abl_{b}_proposed_softmaxhead", fusion_model(sl, head="softmax"), X, X_aug=Xa)
    evaluate(f"abl_{b}_proposed_nomodalitydrop", fusion_model(sl, p_modality=0.0), X, X_aug=Xa)
    for k in (32, 64, 256):
        evaluate(f"abl_{b}_proposed_k{k}", fusion_model(sl, k=k), X, X_aug=Xa)
    for br in (("leaf", "hand"), ("tiles", "hand"), ("leaf", "tiles")):
        X2, sl2, Xa2 = proposed_inputs(b, branches=br)
        evaluate(f"abl_{b}_proposed_{'+'.join(br)}", fusion_model(sl2), X2, X_aug=Xa2)
        if len(br) == 2:
            evaluate(f"abl_{b}_scalar_{'+'.join(br)}", fusion_model(sl2, mode="scalar"), X2, X_aug=Xa2)
    for g in ("col", "glcm", "lbp", "gab", "ven", "mor"):
        keep = [x for x in ("col", "glcm", "lbp", "gab", "ven", "mor") if x != g]
        X2, sl2, Xa2 = proposed_inputs(b, hand_groups=keep)
        evaluate(f"abl_{b}_proposed_minus_{g}", fusion_model(sl2), X2, X_aug=Xa2)


def run_classifiers(b="resnet50"):
    """Classical classifiers on the learned 128-D fused embedding (as in the original manuscript)."""
    X, sl, Xa = proposed_inputs(b)
    heads = {"knn": lambda: KNeighborsClassifier(5), "rf": lambda: RandomForestClassifier(300, random_state=0),
             "dt": lambda: DecisionTreeClassifier(random_state=0), "nb": GaussianNB}
    for name, h in heads.items():
        evaluate(f"clf_{b}_{name}", lambda h=h: EmbedThen(FusionSVM(slices=sl, head="softmax"), h()), X, X_aug=Xa)


class EmbedThen:
    """Train the fusion network, then fit any classifier on its 128-D embedding."""

    def __init__(self, fusion, clf):
        self.fusion, self.clf = fusion, clf

    def fit(self, X, y):
        self.fusion.fit(X, y); self.clf.fit(self.fusion.embed(X), y); return self

    def predict(self, X):
        return self.clf.predict(self.fusion.embed(X))

    def predict_proba(self, X):
        return self.clf.predict_proba(self.fusion.embed(X))


BOARD = G.board.values


def evaluate_custom(name, make_model, X, splits, grid=None, extra=None, force=False):
    """Like evaluate() but with explicit splits (and X possibly a callable returning per-split matrices)."""
    path = RUNS / f"{name}.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    t0 = time.time()
    res = run_cv(make_model, X, y, splits, groups=groups, param_grid=grid)
    out = {"name": name, "folds": res["folds"], "wall_s": time.time() - t0, **(extra or {})}
    np.savez_compressed(RUNS / f"{name}_oof.npz", **{f"pred_s{s}": o["pred"] for s, o in res["oof"].items()},
                        **{f"score_s{s}": o["score"] for s, o in res["oof"].items()})
    path.write_text(json.dumps(out, default=float))
    msg = "  ".join(f"[{f['fold']}] acc {100*f['acc']:.2f} balacc {100*f['bal_acc']:.2f} F1 {100*f['macro_f1']:.2f}"
                    for f in res["folds"][:2])
    print(f"{name:45s} {msg}" if len(res["folds"]) <= 2 else
          f"{name:45s} acc {100*np.mean([f['acc'] for f in res['folds']]):.2f}", flush=True)
    return out


def configs(b):
    """Representations compared in the robustness tests: (name, factory, X builder, grid, uses-aug)."""
    X, sl, Xa = proposed_inputs(b)
    X2, sl2, Xa2 = proposed_inputs(b, branches=("leaf", "tiles"))
    return [("deep_full", svm_pipe, deep(b, "full"), SVM_GRID, None),
            ("legacy34", svm_pipe, LEGACY, SVM_GRID, None),
            ("deep_leaf", svm_pipe, deep(b, "leaf"), SVM_GRID, None),
            ("hand164", svm_pipe, hand(), SVM_GRID, None),
            ("proposed", lambda: FusionSVM(slices=sl), X, None, Xa),
            ("leaf+tiles", lambda: FusionSVM(slices=sl2), X2, None, Xa2)]


def run_crossboard(b="resnet50"):
    """Train on one board type, test on the other (both directions)."""
    tr0, tr1 = np.nonzero(BOARD == 0)[0], np.nonzero(BOARD == 1)[0]
    splits = [(0, 0, tr0, tr1), (0, 1, tr1, tr0)]  # fold 0: stripes->squares, fold 1: squares->stripes
    for name, mk, X, grid, Xa in configs(b):
        if Xa is not None:
            Xc = lambda tr, te, X=X, Xa=Xa: (X[tr], X[te], Xa[:, tr].reshape(-1, X.shape[1]))
        else:
            Xc = X
        evaluate_custom(f"xboard_{b}_{name}", mk, Xc, splits, grid=grid)


def swap_inputs(b):
    """Feature matrices of the background-swapped composites, aligned with configs(b)."""
    S = dict(np.load(CACHE / f"swap_{b}.npz"))
    Hs = pd.read_parquet(CACHE / "swap_handcrafted.parquet")[HAND.columns].values
    Ls = pd.read_parquet(CACHE / "swap_handcrafted_legacy.parquet").values
    Xs, _ = stack(S["leaf"], S["tiles"], Hs)
    Xs2, _ = stack(S["leaf"], S["tiles"])
    return {"deep_full": S["full"], "legacy34": Ls, "deep_leaf": S["leaf"], "hand164": Hs, "proposed": Xs,
            "leaf+tiles": Xs2}


def run_swap(b="resnet50"):
    """Train on original images of the training fold, test on background-swapped composites of the test fold."""
    sw = swap_inputs(b)
    partner_y = pd.read_csv(CACHE / "swap_meta.csv").partner_y.values
    splits = repeated_group_splits(y, groups, seeds=(0, 1))
    for name, mk, X, grid, Xa in configs(b):
        Xs = sw[name]
        Xc = lambda tr, te, X=X, Xs=Xs, Xa=Xa: (X[tr], Xs[te], None if Xa is None else Xa[:, tr].reshape(-1, X.shape[1]))
        out = evaluate_custom(f"swap_{b}_{name}", mk, Xc, splits, grid=grid)
        oof = np.load(RUNS / f"swap_{b}_{name}_oof.npz")
        follow = np.mean([np.mean(oof[f"pred_s{s}"] == partner_y) for s in (0, 1)])
        acc = np.mean([np.mean(oof[f"pred_s{s}"] == y) for s in (0, 1)])
        out["swap_acc"], out["follow_rate"] = float(acc), float(follow)
        (RUNS / f"swap_{b}_{name}.json").write_text(json.dumps(out, default=float))
        print(f"   -> swap accuracy {100*acc:.2f}%, prediction follows the pasted board's variety {100*follow:.2f}%")


def run_protocols(b="dinov2_s"):
    """Proposed vs full-image deep baseline under random, duplicate-only-grouped and sequence-grouped CV."""
    X, sl, Xa = proposed_inputs(b)
    for proto in ("random", "groupdup", "group"):
        evaluate(f"proto_{proto}_{b}_deep_full", svm_pipe, deep(b, "full"), grid=SVM_GRID, protocol=proto)
        evaluate(f"proto_{proto}_{b}_proposed", fusion_model(sl), X, X_aug=Xa, protocol=proto)


def run_dedup(b="dinov2_s"):
    """Sensitivity: drop the byte-identical copies (keep the first of each SHA-256) and re-evaluate."""
    keep = np.nonzero(~df.duplicated("hash_sha256").values)[0]
    yk, gk = y[keep], groups[keep]
    X, sl, Xa = proposed_inputs(b)
    for name, mk, Xm, Xam, grid in (("deep_full", svm_pipe, deep(b, "full"), None, SVM_GRID),
                                     ("proposed", fusion_model(sl), X, Xa, None)):
        path = RUNS / f"dedup_{b}_{name}.json"
        if path.exists() or not selected(path.stem):
            continue
        splits = repeated_group_splits(yk, gk, seeds=SEEDS)
        res = run_cv(mk, Xm[keep], yk, splits, groups=gk, X_aug=None if Xam is None else Xam[:, keep], param_grid=grid)
        out = {"name": path.stem, "n_images": int(len(keep)), "summary": summarise(res, yk), "folds": res["folds"]}
        path.write_text(json.dumps(out, default=float))
        print(f"{path.stem:45s} acc {100*out['summary']['acc']['mean']:.2f}±{100*out['summary']['acc']['std']:.2f}", flush=True)


def run_backbones():
    # resnet50 and dinov2_s are covered by the cmp_* runs of run_compare
    for b in ("effnet_b0", "convnext_t", "mobilenetv3"):
        evaluate(f"bb_{b}_deep_full", svm_pipe, deep(b, "full"), grid=SVM_GRID)
        X, sl, Xa = proposed_inputs(b)
        evaluate(f"bb_{b}_proposed", fusion_model(sl), X, X_aug=Xa)


if __name__ == "__main__":
    {"leakage": run_leakage, "audit": run_audit, "compare": run_compare, "ablation": run_ablation,
     "classifiers": run_classifiers, "backbones": run_backbones, "crossboard": run_crossboard, "swap": run_swap,
     "protocols": run_protocols, "dedup": run_dedup}[sys.argv[1]](*sys.argv[2:])
