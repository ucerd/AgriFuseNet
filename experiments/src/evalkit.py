"""Leakage-free evaluation: repeated stratified group k-fold, train-only augmentation and tuning, statistics."""
import time
import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score, cohen_kappa_score,
                             precision_recall_fscore_support, roc_auc_score, average_precision_score,
                             confusion_matrix)
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, GroupKFold
from sklearn.preprocessing import label_binarize

N_CLASSES = 5


def repeated_group_splits(y, groups, n_splits=5, seeds=(0, 1, 2, 3, 4)):
    out = []
    for s in seeds:
        for k, (tr, te) in enumerate(StratifiedGroupKFold(n_splits, shuffle=True, random_state=s).split(y, y, groups)):
            out.append((s, k, tr, te))
    return out


def random_splits(y, n_splits=5, seeds=(0, 1, 2, 3, 4)):
    """Ordinary stratified k-fold that ignores leaf identity (the leaky protocol), for comparison."""
    from sklearn.model_selection import StratifiedKFold
    return [(s, k, tr, te) for s in seeds
            for k, (tr, te) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=s).split(y, y))]


def scores_of(model, X):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except AttributeError:
            pass
    # raw one-vs-rest margins: each column ranks "class c vs rest", which is all ROC/PR need
    return model.decision_function(X)


def fold_metrics(y, pred, score):
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=range(N_CLASSES), zero_division=0)
    Y = label_binarize(y, classes=range(N_CLASSES))
    return {"acc": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
            "bal_acc": balanced_accuracy_score(y, pred), "kappa": cohen_kappa_score(y, pred),
            "macro_auc": roc_auc_score(Y, score, average="macro"),
            "macro_ap": average_precision_score(Y, score, average="macro"),
            "per_class_f1": f.tolist(), "per_class_precision": p.tolist(), "per_class_recall": r.tolist()}


def run_cv(make_model, X, y, splits, groups=None, X_aug=None, param_grid=None, inner_splits=3, verbose=False):
    """Evaluate a model factory over the given outer splits.

    make_model() -> fresh estimator. X may be an array or a callable(train_idx, test_idx) -> (Xtr, Xte)
    for methods whose features are themselves learned on the training fold (e.g. the fusion network).
    X_aug: optional (n_aug, N, D) features of augmented copies; only rows of training images are used.
    param_grid: tuned by an inner GroupKFold on the training fold (non-augmented rows), then refit.
    Returns dict with per-fold metrics, pooled out-of-fold scores per seed, and timings.
    """
    folds, oof = [], {}
    for s, k, tr, te in splits:
        t0 = time.perf_counter()
        if callable(X):
            Xtr, Xte, Xtr_aug = X(tr, te)
        else:
            Xtr, Xte = X[tr], X[te]
            Xtr_aug = None if X_aug is None else X_aug[:, tr].reshape(-1, X.shape[1])
        ytr = y[tr]
        model = make_model()
        best = None
        if param_grid:
            gs = GridSearchCV(model, param_grid, cv=GroupKFold(inner_splits), scoring="f1_macro", n_jobs=4)
            gs.fit(Xtr, ytr, groups=groups[tr])
            best = gs.best_params_
            model = clone(model).set_params(**best)
        if Xtr_aug is not None:
            n_aug = len(Xtr_aug) // len(tr)
            model.fit(np.vstack([Xtr, Xtr_aug]), np.concatenate([ytr, np.tile(ytr, n_aug)]))
        else:
            model.fit(Xtr, ytr)
        t1 = time.perf_counter()
        pred = model.predict(Xte)
        sc = scores_of(model, Xte)
        t2 = time.perf_counter()
        m = fold_metrics(y[te], pred, sc)
        m.update(seed=s, fold=k, best=best, fit_s=t1 - t0, predict_ms_per_img=1000 * (t2 - t1) / len(te))
        folds.append(m)
        o = oof.setdefault(s, {"score": np.zeros((len(y), N_CLASSES)), "pred": np.zeros(len(y), int)})
        o["score"][te], o["pred"][te] = sc, pred
        if verbose:
            print(f"  seed {s} fold {k}: acc {m['acc']:.4f} f1 {m['macro_f1']:.4f} {best or ''}", flush=True)
    return {"folds": folds, "oof": oof}


def summarise(res, y):
    F = res["folds"]
    out = {}
    for key in ("acc", "macro_f1", "bal_acc", "kappa", "macro_auc", "macro_ap", "fit_s", "predict_ms_per_img"):
        v = np.array([f[key] for f in F])
        ci = stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else (v[0], v[0])
        out[key] = {"mean": v.mean(), "std": v.std(ddof=1) if len(v) > 1 else 0.0, "ci95": [float(ci[0]), float(ci[1])]}
    # per-seed pooled confusion matrices (every image is tested exactly once per seed)
    cms = [confusion_matrix(y, o["pred"], labels=range(N_CLASSES)) for o in res["oof"].values()]
    out["confusion_mean"] = np.mean(cms, 0).tolist()
    out["per_class_f1_mean"] = np.mean([f["per_class_f1"] for f in F], 0).tolist()
    out["per_class_recall_mean"] = np.mean([f["per_class_recall"] for f in F], 0).tolist()
    out["per_class_precision_mean"] = np.mean([f["per_class_precision"] for f in F], 0).tolist()
    return out


def corrected_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated k-fold paired scores."""
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), J - 1)
    return float(t), float(p)
