"""Paired comparison of two leaf-blade runs with the corrected resampled t-test (Nadeau & Bengio, 2003).

python scripts/compare_runs.py RUN_A.json RUN_B.json [--metric acc]
Splits are paired by (seed, fold); both runs must use the same protocol and seeds.
"""
import argparse
import json

import numpy as np

from agrifusenet.evaluation import corrected_ttest


def per_split(path, metric):
    folds = json.loads(open(path).read())["folds"]
    return {(f["seed"], f["fold"]): f[metric] for f in folds}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--metric", default="acc")
    ap.add_argument("--test-train-ratio", type=float, default=0.25, help="n_test / n_train (0.25 for 5-fold CV)")
    args = ap.parse_args()
    A, B = per_split(args.a, args.metric), per_split(args.b, args.metric)
    keys = sorted(set(A) & set(B))
    a, b = np.array([A[k] for k in keys]), np.array([B[k] for k in keys])
    t, p = corrected_ttest(a, b, args.test_train_ratio)
    print(f"{len(keys)} paired splits | {args.metric}: A {100 * a.mean():.2f}, B {100 * b.mean():.2f}, "
          f"A better in {int((a > b).sum())}/{len(keys)} | t = {t:.3f}, p = {p:.4f}")


if __name__ == "__main__":
    main()
