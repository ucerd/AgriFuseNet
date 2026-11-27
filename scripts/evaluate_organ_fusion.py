"""Multi-organ variety identification: organ-set fusion versus late and early fusion on identical specimens.

Organ representations (no background pixel reaches any encoder)
  leaf blade : the leaf-blade encoder branches [masked blade | blade tiles | botanical descriptors]
               (--leaf-encoder agrifusenet) or the DINOv2 feature of the SAM 3 leaf crop (--leaf-encoder sam3-crop)
  leaf top, bud, cross-section : DINOv2 features of the SAM 3 organ crops
Per split, every method is evaluated on the same test specimens and on every non-empty subset of organs.

python scripts/evaluate_organ_fusion.py --data-root DATA --protocol grouped --seeds 0,1,2,3,4
python scripts/evaluate_organ_fusion.py --data-root DATA --protocol session
-> <out-dir>/results/organ_fusion/<name>.csv (one row per split, method and organ subset)
"""
import argparse
import itertools
import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from agrifusenet.config import ORGANS, add_path_args, paths_from_args
from agrifusenet.data import load_groups
from agrifusenet.evaluation import corrected_ttest
from agrifusenet.organ_fusion import (organ_splits, draw_specimens, fit_set_model, predict_set, organ_scores,
                                      late_fusion, early_fusion, N_CLASSES)

METHODS = ("late_sum", "early_concat", "set_mean", "set_attn")


def load_organs(paths, leaf_encoder="agrifusenet", descriptors=True):
    """Per organ: X (n, d), Xa (n_aug, n, d), y, leakage groups g, capture period, branch slices."""
    crops = np.load(paths.features / "organ_crops_dinov2_s.npz")
    crow = {p: i for i, p in enumerate(crops["relative_path"])}
    data = {}
    for o in ORGANS:
        G = load_groups(paths, o)
        if o == "leafwidth" and leaf_encoder == "agrifusenet":
            D = np.load(paths.features / "leaf_dinov2_s.npz")
            assert (D["relative_path"] == G.relative_path.values).all()
            blocks, aug = [D["leaf"], D["tiles"]], [D["aug_leaf"], D["aug_tiles"]]
            if descriptors:
                H = pd.read_parquet(paths.features / "leaf_descriptors.parquet").loc[G.relative_path].values
                blocks.append(H)
                aug.append(np.repeat(H[None], D["aug_leaf"].shape[0], 0))
            X = np.hstack(blocks).astype(np.float32)
            Xa = np.concatenate(aug, 2).astype(np.float32)
            edges = np.cumsum([0] + [b.shape[1] for b in blocks])
            slices = [slice(a, b) for a, b in zip(edges[:-1], edges[1:])]
        else:
            ix = np.array([crow[p] for p in G.relative_path])
            X, Xa = crops["crop"][ix], crops["aug_crop"][:, ix]
            slices = [slice(0, X.shape[1])]
        data[o] = dict(X=X, Xa=Xa, y=G.y.values, g=G.group.values, period=G.period.values, slices=slices)
    return data


def summary(df, reference="set_mean"):
    subsets = {"all organs": "+".join(ORGANS), "leaf views": "leafwidth+leavePattern"}
    lines = []
    for label, sub in subsets.items():
        d = df[df.subset == sub]
        ref = d[d.method == reference].sort_values(["seed", "fold"]).acc.values
        for m, g in d.groupby("method"):
            a = g.sort_values(["seed", "fold"]).acc.values
            p = corrected_ttest(ref, a)[1] if m != reference and len(a) == len(ref) and len(a) > 2 else float("nan")
            lines.append(f"{label:10s} {m:13s} accuracy {100 * a.mean():6.2f} +- {100 * a.std(ddof=1) if len(a) > 1 else 0:5.2f}"
                         + (f"  p vs {reference} = {p:.3f}" if not np.isnan(p) else ""))
    return "\n".join(lines)


def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--protocol", choices=["grouped", "session"], default="grouped")
    ap.add_argument("--seeds", default="0", help="grouped protocol: comma-separated seeds of the 5-fold splits")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--leaf-encoder", choices=["agrifusenet", "sam3-crop"], default="agrifusenet")
    ap.add_argument("--no-descriptors", action="store_true", help="leaf-blade encoder without botanical descriptors")
    ap.add_argument("--no-augmentation", action="store_true")
    ap.add_argument("--organ-dropout", type=float, default=0.3)
    ap.add_argument("--pairing", choices=["random", "period"], default="random")
    ap.add_argument("--n-test", type=int, default=200, help="test specimens per variety and split")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    paths = paths_from_args(args)
    methods = args.methods.split(",")
    data = load_organs(paths, args.leaf_encoder, descriptors=not args.no_descriptors)
    subsets = {"+".join(c): set(c) for r in range(1, len(ORGANS) + 1) for c in itertools.combinations(ORGANS, r)}
    early_subsets = {"+".join(ORGANS), "leafwidth+leavePattern"}
    seeds = [int(s) for s in args.seeds.split(",")] if args.protocol == "grouped" else [0]
    rows, t0 = [], time.time()
    for s in seeds:
        for k, sp in enumerate(organ_splits(data, args.protocol, s)):
            tr, te = {o: sp[o][0] for o in ORGANS}, {o: sp[o][1] for o in ORGANS}
            rng = np.random.default_rng(1000 + 10 * s + k)
            spec, ys = draw_specimens(data, te, args.n_test, rng, args.pairing)
            preds = {}
            if "late_sum" in methods:
                Z = organ_scores(data, tr, not args.no_augmentation)
                for name, sub in subsets.items():
                    preds[("late_sum", name)] = late_fusion(Z, spec, ORGANS, sub)
            if "early_concat" in methods:
                for name in early_subsets:
                    preds[("early_concat", name)] = early_fusion(data, tr, spec, subsets[name],
                                                                 np.random.default_rng(s * 7 + k))
            for pool in ("mean", "attn"):
                if f"set_{pool}" in methods:
                    net, sc = fit_set_model(data, tr, pool, seed=s, use_aug=not args.no_augmentation,
                                            p_organ_drop=args.organ_dropout)
                    for name, sub in subsets.items():
                        preds[(f"set_{pool}", name)] = predict_set(net, sc, data, spec, sub)
            for (m, name), p in preds.items():
                rows.append({"protocol": args.protocol, "pairing": args.pairing, "seed": s, "fold": k, "method": m,
                             "subset": name, "n_organs": len(subsets[name]), "acc": float((p == ys).mean()),
                             "per_class_recall": [float((p[ys == c] == c).mean()) for c in range(N_CLASSES)],
                             "confusion": confusion_matrix(ys, p, labels=list(range(N_CLASSES))).tolist()})
            allk = "+".join(ORGANS)
            print(f"{args.protocol} seed {s} split {k}, all organs: " +
                  " ".join(f"{r['method']} {100 * r['acc']:.1f}" for r in rows[-len(preds):] if r["subset"] == allk) +
                  f" ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    name = args.name or f"{args.protocol}_{args.leaf_encoder}_{args.pairing}_seeds{args.seeds.replace(',', '-')}"
    folder = paths.results / "organ_fusion"
    folder.mkdir(exist_ok=True)
    df.to_csv(folder / f"{name}.csv", index=False)
    print(summary(df))


if __name__ == "__main__":
    main()
