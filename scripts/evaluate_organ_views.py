"""Background and session diagnostics for single organs: balanced accuracy of four views of each organ image.

  frame      : the full photograph
  corners    : the four 112-px corner patches only
  background : the photograph with the (dilated) SAM 3 organ mask greyed out
  crop       : the background-free SAM 3 organ crop
Each view is classified by an RBF SVM (C = 10) under grouped cross-validation (5-fold, --seeds) and session-level
cross-validation (leave one capture period out). Augmented copies are used for the frame and crop views.

python scripts/evaluate_organ_views.py --data-root DATA  ->  <out-dir>/results/organ_views.csv
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from agrifusenet.config import ORGANS, VARIETIES, add_path_args, paths_from_args
from agrifusenet.data import load_groups

VIEWS = {"frame": ("frames", "frame", "aug_frame"), "corners": ("frames", "corners", None),
         "background": ("crops", "background", None), "crop": ("crops", "crop", "aug_crop")}


def out_of_split_predictions(X, Xa, y, splits):
    pred = np.zeros(len(y), int)
    for tr, te in splits:
        Xtr, ytr = X[tr], y[tr]
        if Xa is not None:
            Xtr = np.vstack([Xtr, Xa[:, tr].reshape(-1, X.shape[1])])
            ytr = np.concatenate([ytr, np.tile(y[tr], Xa.shape[0])])
        m = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")).fit(Xtr, ytr)
        pred[te] = m.classes_[m.decision_function(X[te]).argmax(1)]
    return pred


def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--categories", nargs="+", default=ORGANS)
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    paths = paths_from_args(args)
    seeds = [int(s) for s in args.seeds.split(",")]
    files = {"frames": np.load(paths.features / "frames_dinov2_s.npz"),
             "crops": np.load(paths.features / "organ_crops_dinov2_s.npz")}
    index = {k: {p: i for i, p in enumerate(f["relative_path"])} for k, f in files.items()}
    rows = []
    for cat in args.categories:
        G = load_groups(paths, cat)
        y, g, period = G.y.values, G.group.values, G.period.values
        for view, (src, key, aug_key) in VIEWS.items():
            ix = np.array([index[src][p] for p in G.relative_path])
            X = files[src][key][ix]
            Xa = files[src][aug_key][:, ix] if aug_key and aug_key in files[src].files else None
            grouped = []
            for s in seeds:
                sp = list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, y, g))
                p = out_of_split_predictions(X, Xa, y, sp)
                grouped.append((accuracy_score(y, p), balanced_accuracy_score(y, p)))
            p = out_of_split_predictions(X, Xa, y, [(np.nonzero(period != k)[0], np.nonzero(period == k)[0])
                                                    for k in np.unique(period)])
            rows.append({"category": cat, "view": view, "n_images": len(y), "n_varieties": len(np.unique(y)),
                         "grouped_acc": np.mean([a for a, _ in grouped]), "grouped_bal_acc": np.mean([b for _, b in grouped]),
                         "session_acc": accuracy_score(y, p), "session_bal_acc": balanced_accuracy_score(y, p)})
            r = rows[-1]
            print(f"{cat:14s} {view:10s} balanced accuracy: grouped {100 * r['grouped_bal_acc']:5.1f}  "
                  f"session {100 * r['session_bal_acc']:5.1f}", flush=True)
    pd.DataFrame(rows).to_csv(paths.results / "organ_views.csv", index=False)


if __name__ == "__main__":
    main()
