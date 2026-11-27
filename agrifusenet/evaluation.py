"""Evaluation protocols, metrics and the corrected resampled t-test.

Protocols
  grouped : repeated stratified group k-fold over leakage groups (default 5 x 5)
  random  : repeated stratified k-fold that ignores the groups (for comparison only)
  session : leave-one-capture-period-out (train on two periods, test on the third)
  boards  : train on one background board, test on the other (both directions)
Augmented copies are only ever used for training images; hyper-parameters are tuned by an inner group k-fold
on the training part of each split.
"""
import time

import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score, cohen_kappa_score,
                             precision_recall_fscore_support, roc_auc_score, average_precision_score,
                             confusion_matrix)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, GridSearchCV, GroupKFold
from sklearn.preprocessing import label_binarize

N_CLASSES = 5
SEEDS = (0, 1, 2, 3, 4)


def grouped_splits(y, groups, n_splits=5, seeds=SEEDS):
    return [(s, k, tr, te) for s in seeds for k, (tr, te) in
            enumerate(StratifiedGroupKFold(n_splits, shuffle=True, random_state=s).split(y, y, groups))]


def random_splits(y, n_splits=5, seeds=SEEDS):
    return [(s, k, tr, te) for s in seeds for k, (tr, te) in
            enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=s).split(y, y))]


def session_splits(periods):
    return [(0, k, np.nonzero(periods != k)[0], np.nonzero(periods == k)[0]) for k in np.unique(periods)]


def board_splits(boards):
    a, b = np.nonzero(boards == 0)[0], np.nonzero(boards == 1)[0]
    return [(0, 0, a, b), (0, 1, b, a)]


def scores_of(model, X):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except AttributeError:
            pass
    return model.decision_function(X)  # one-vs-rest margins rank every class against the rest


def fold_metrics(y, pred, score):
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=range(N_CLASSES), zero_division=0)
    Y = label_binarize(y, classes=range(N_CLASSES))
    present = Y.sum(0) > 0
    return {"acc": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
            "bal_acc": balanced_accuracy_score(y, pred), "kappa": cohen_kappa_score(y, pred),
            "macro_auc": roc_auc_score(Y[:, present], score[:, present], average="macro") if present.sum() > 1 else float("nan"),
            "macro_ap": average_precision_score(Y[:, present], score[:, present], average="macro"),
            "per_class_f1": f.tolist(), "per_class_precision": p.tolist(), "per_class_recall": r.tolist()}


def run_cv(make_model, X, y, splits, groups=None, X_aug=None, param_grid=None, inner_splits=3, X_test=None):
    """Evaluate a model factory over the outer splits.

    X      : (N, D) features. X_test, if given, replaces X for the test images (e.g. counterfactual images).
    X_aug  : optional (n_aug, N, D) features of augmented copies; only rows of training images are used.
    param_grid : tuned by an inner GroupKFold on the training images, then the model is refitted.
    Returns per-split metrics and, per seed, the out-of-split scores and predictions.
    """
    X_test = X if X_test is None else X_test
    folds, oof = [], {}
    for s, k, tr, te in splits:
        t0 = time.perf_counter()
        Xtr, ytr = X[tr], y[tr]
        model = make_model()
        best = None
        if param_grid:
            gs = GridSearchCV(model, param_grid, cv=GroupKFold(inner_splits), scoring="f1_macro", n_jobs=4)
            gs.fit(Xtr, ytr, groups=groups[tr])
            best = gs.best_params_
            model = clone(model).set_params(**best)
        if X_aug is not None:
            n_aug = X_aug.shape[0]
            model.fit(np.vstack([Xtr, X_aug[:, tr].reshape(-1, X.shape[1])]), np.concatenate([ytr, np.tile(ytr, n_aug)]))
        else:
            model.fit(Xtr, ytr)
        t1 = time.perf_counter()
        pred = model.predict(X_test[te])
        sc = scores_of(model, X_test[te])
        t2 = time.perf_counter()
        m = fold_metrics(y[te], pred, sc)
        m.update(seed=s, fold=k, best=best, fit_s=t1 - t0, predict_ms_per_img=1000 * (t2 - t1) / len(te))
        folds.append(m)
        o = oof.setdefault(s, {"score": np.zeros((len(y), N_CLASSES)), "pred": np.full(len(y), -1)})
        o["score"][te], o["pred"][te] = sc, pred
    return {"folds": folds, "oof": oof}


def summarise(res):
    F = res["folds"]
    out = {}
    for key in ("acc", "macro_f1", "bal_acc", "kappa", "macro_auc", "macro_ap", "fit_s", "predict_ms_per_img"):
        v = np.array([f[key] for f in F], float)
        out[key] = {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0}
    out["per_class_recall_mean"] = np.mean([f["per_class_recall"] for f in F], 0).tolist()
    out["per_class_f1_mean"] = np.mean([f["per_class_f1"] for f in F], 0).tolist()
    return out


def pooled_confusion(res, y):
    """Mean over seeds of the confusion matrix of the out-of-split predictions."""
    cms = [confusion_matrix(y[o["pred"] >= 0], o["pred"][o["pred"] >= 0], labels=range(N_CLASSES))
           for o in res["oof"].values()]
    return np.mean(cms, 0)


def corrected_ttest(a, b, test_train_ratio=0.25):
    """Nadeau & Bengio (2003) corrected resampled t-test for paired scores of repeated k-fold splits
    (test_train_ratio = n_test / n_train, 0.25 for 5-fold cross-validation). Returns (t, p)."""
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    if J < 2 or d.var(ddof=1) == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / J + test_train_ratio) * d.var(ddof=1))
    return float(t), float(2 * stats.t.sf(abs(t), J - 1))
