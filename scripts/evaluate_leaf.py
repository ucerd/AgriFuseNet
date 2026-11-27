"""Evaluate a leaf-blade representation and classifier under a chosen protocol.

Representations (--inputs, concatenated in the given order)
  full, leaf, background, background_cells, tiles : frozen backbone features of the image views
  descriptors : 164-D botanical descriptors (--descriptor-groups selects groups)
  baseline34  : 34-D whole-image descriptor
Models (--model)
  svm         : standardisation [+ PCA / LDA] + SVM; C and kernel tuned by inner group k-fold
  encoder     : leaf-blade encoder (channel-wise attention fusion, one branch per input) + SVM or other head
  scalar-gate : two-branch scalar-gated fusion + SVM
Protocols (--protocol): grouped, duplicates, random, session, boards.
--test-on swapped evaluates models trained on the original images on background-swapped test images.

Examples
  leaf-blade encoder:  --inputs leaf tiles descriptors --model encoder
  full-image baseline: --inputs full --model svm
  background probe:    --inputs background_cells --model svm
Writes <out-dir>/results/leaf/<name>.json (summary and per-split metrics) and <name>_predictions.npz.
"""
import argparse
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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from agrifusenet.backbones import BACKBONES
from agrifusenet.config import add_path_args, paths_from_args
from agrifusenet.data import read_metadata, category, load_groups
from agrifusenet.descriptors import GROUPS
from agrifusenet.evaluation import (grouped_splits, random_splits, session_splits, board_splits, run_cv, summarise,
                                    pooled_confusion)
from agrifusenet.leaf_encoder import LeafEncoder, EmbeddingClassifier

warnings.filterwarnings("ignore")
DEEP = ("full", "leaf", "background", "background_cells", "tiles")
SVM_GRID = {"svc__C": [1, 10, 100], "svc__kernel": ["rbf", "linear"]}
HEADS = {"knn": lambda: KNeighborsClassifier(5), "rf": lambda: RandomForestClassifier(300, random_state=0),
         "dt": lambda: DecisionTreeClassifier(random_state=0), "nb": GaussianNB}


def load_inputs(paths, backbone, inputs, descriptor_groups, swapped=False):
    """Feature blocks (N, D) and augmented blocks (n_aug, N, D) or None for inputs without augmentation."""
    prefix = "swap" if swapped else "leaf"
    deep = np.load(paths.features / f"{prefix}_{backbone}.npz") if set(inputs) & set(DEEP) else None
    blocks, aug = [], []
    for name in inputs:
        if name in DEEP:
            blocks.append(deep[name])
            aug.append(deep["aug_" + name] if (not swapped and "aug_" + name in deep.files) else None)
        elif name == "descriptors":
            H = pd.read_parquet(paths.features / f"{prefix}_descriptors.parquet")
            blocks.append(H.loc[:, H.columns.str.split("_").str[0].isin(descriptor_groups)].values)
            aug.append("static")
        elif name == "baseline34":
            blocks.append(pd.read_parquet(paths.features / f"{prefix}_baseline34.parquet").values)
            aug.append("static")
    return [b.astype(np.float32) for b in blocks], aug


def stack(blocks):
    edges = np.cumsum([0] + [b.shape[1] for b in blocks])
    return np.hstack(blocks), [slice(a, b) for a, b in zip(edges[:-1], edges[1:])]


def stack_aug(blocks, aug):
    """Augmented copies of the deep inputs; static inputs (descriptors) are repeated for each copy."""
    if any(a is None for a in aug) or all(isinstance(a, str) for a in aug):
        return None
    n = max(a.shape[0] for a in aug if not isinstance(a, str))
    return np.concatenate([np.repeat(b[None], n, 0) if isinstance(a, str) else a for b, a in zip(blocks, aug)],
                          2).astype(np.float32)


def model_factory(args, slices):
    if args.model == "svm":
        def make():
            steps = [StandardScaler()]
            if args.reduce == "pca128":
                steps.append(PCA(128, random_state=0))
            elif args.reduce == "lda":
                steps.append(LinearDiscriminantAnalysis())
            return make_pipeline(*steps, SVC(gamma="scale"))
        return make, SVM_GRID
    kw = dict(slices=slices, k=args.dim, mode="scalar" if args.model == "scalar-gate" else "channel",
              p_drop=args.dropout, label_smoothing=args.label_smoothing,
              p_modality=0.0 if args.no_modality_dropout else 0.2)
    if args.head in HEADS:
        return (lambda: EmbeddingClassifier(LeafEncoder(head="softmax", **kw), HEADS[args.head]())), None
    return (lambda: LeafEncoder(head=args.head, **kw)), None


def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--backbone", choices=BACKBONES, default="dinov2_s")
    ap.add_argument("--inputs", nargs="+", default=["leaf", "tiles", "descriptors"],
                    choices=list(DEEP) + ["descriptors", "baseline34"])
    ap.add_argument("--model", choices=["svm", "encoder", "scalar-gate"], default="encoder")
    ap.add_argument("--reduce", choices=["none", "pca128", "lda"], default="none", help="svm only")
    ap.add_argument("--head", choices=["svm", "softmax"] + list(HEADS), default="svm", help="encoder only")
    ap.add_argument("--dim", type=int, default=128, help="dimension of the fused code")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--no-modality-dropout", action="store_true")
    ap.add_argument("--no-augmentation", action="store_true")
    ap.add_argument("--descriptor-groups", nargs="+", default=list(GROUPS), choices=GROUPS)
    ap.add_argument("--protocol", choices=["grouped", "duplicates", "random", "session", "boards"], default="grouped")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--drop-exact-duplicates", action="store_true", help="keep one file per SHA-256")
    ap.add_argument("--test-on", choices=["original", "swapped"], default="original")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    paths = paths_from_args(args)

    df = category(read_metadata(paths), "leafwidth")
    G = load_groups(paths, "leafwidth")
    assert (G.relative_path.values == df.relative_path.values).all()
    y, groups = df.y.values, G.group.values
    blocks, aug = load_inputs(paths, args.backbone, args.inputs, args.descriptor_groups)
    X, slices = stack(blocks)
    X_aug = None if (args.model == "svm" or args.no_augmentation) else stack_aug(blocks, aug)
    X_test = None
    if args.test_on == "swapped":
        X_test, _ = stack(load_inputs(paths, args.backbone, args.inputs, args.descriptor_groups, swapped=True)[0])

    keep = np.arange(len(y))
    if args.drop_exact_duplicates:
        keep = np.nonzero(~df.duplicated("hash_sha256").values)[0]
    X, y, groups = X[keep], y[keep], groups[keep]
    X_aug = None if X_aug is None else X_aug[:, keep]
    X_test = None if X_test is None else X_test[keep]

    seeds = tuple(int(s) for s in args.seeds.split(","))
    splits = {"grouped": lambda: grouped_splits(y, groups, seeds=seeds),
              "duplicates": lambda: grouped_splits(y, G.duplicate_group.values[keep], seeds=seeds),
              "random": lambda: random_splits(y, seeds=seeds),
              "session": lambda: session_splits(G.period.values[keep]),
              "boards": lambda: board_splits(G.board.values[keep])}[args.protocol]()

    make, grid = model_factory(args, slices)
    t0 = time.time()
    res = run_cv(make, X, y, splits, groups=groups, X_aug=X_aug, param_grid=grid, X_test=X_test)
    name = args.name or "_".join([args.backbone, args.model, "+".join(args.inputs), args.protocol])
    out = {"name": name, "args": vars(args), "n_images": int(len(y)), "summary": summarise(res),
           "confusion": pooled_confusion(res, y).tolist(), "folds": res["folds"], "wall_s": time.time() - t0}
    if args.test_on == "swapped":
        partner_y = pd.read_csv(paths.swap / "pairs.csv").partner_y.values[keep]
        preds = [o["pred"] for o in res["oof"].values()]
        out["swap_accuracy"] = float(np.mean([np.mean(p == y) for p in preds]))
        out["follow_rate"] = float(np.mean([np.mean(p == partner_y) for p in preds]))  # follows the pasted board
    folder = paths.results / "leaf"
    folder.mkdir(exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps(out, indent=1, default=float))
    np.savez_compressed(folder / f"{name}_predictions.npz", **{f"score_seed{s}": o["score"] for s, o in res["oof"].items()},
                        **{f"pred_seed{s}": o["pred"] for s, o in res["oof"].items()})
    s = out["summary"]
    msg = f"{name}: accuracy {100 * s['acc']['mean']:.2f} +- {100 * s['acc']['std']:.2f}, " \
          f"macro-F1 {100 * s['macro_f1']['mean']:.2f}, balanced accuracy {100 * s['bal_acc']['mean']:.2f}"
    if "follow_rate" in out:
        msg += f", swap accuracy {100 * out['swap_accuracy']:.2f}, follow rate {100 * out['follow_rate']:.2f}"
    print(msg)


if __name__ == "__main__":
    main()
