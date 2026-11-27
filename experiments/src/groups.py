"""Leaf-identity groups (for leakage-free splits) and background-board clusters (for the shortcut audit).

python groups.py analyse      -> similarity statistics used to choose the threshold
python groups.py build TAU    -> cache/groups.csv with columns group, board
"""
import sys
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from common import load_manifest, CACHE, RESULTS, CLASSES


def leaf_embeddings():
    d = np.load(CACHE / "deep_dinov2_s.npz")
    return normalize(d["leaf"]), normalize(d["bgpatch"])


def analyse(df, E):
    S = E @ E.T
    np.fill_diagonal(S, -1)
    y = df.y.values
    same = y[:, None] == y[None]
    nn_same = np.where(same, S, -1).max(1)
    nn_other = np.where(~same, S, -1).max(1)
    idx = df.relative_path.str.extract(r"img_(\d+)")[0].astype(int).values
    cons = [S[i, j] for i in range(len(df)) for j in range(i + 1, len(df))
            if y[i] == y[j] and abs(idx[i] - idx[j]) == 1]
    rnd = S[np.triu_indices(len(df), 1)][same[np.triu_indices(len(df), 1)]]
    q = lambda v: {p: round(float(np.percentile(v, p)), 4) for p in (5, 25, 50, 75, 95)}
    print("NN similarity same class ", q(nn_same))
    print("NN similarity other class", q(nn_other))
    print("consecutive-index pairs  ", q(np.array(cons)))
    print("random same-class pairs  ", q(rnd))
    for tau, block in ((0.95, None), (0.97, None), (0.985, None), (0.985, 5), (0.985, 10), (0.985, 20), (0.97, 10)):
        g = components(df, S, tau, block)
        sizes = pd.Series(g).value_counts()
        per_class = pd.Series(g).groupby(df.variety.values).nunique().to_dict()
        print(f"tau {tau} block {block}: {sizes.size} groups, max size {sizes.max()}, singletons {(sizes == 1).sum()}, per class {per_class}")


def components(df, S, tau, block=None):
    """Union-find over exact duplicates, same-class pairs with similarity >= tau and, when block is
    given, contiguous capture-sequence blocks of `block` consecutive images within each class."""
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    y = df.y.values
    for _, ix in df.groupby("hash_sha256").groups.items():
        ix = list(ix)
        for j in ix[1:]:
            union(ix[0], j)
    ii, jj = np.nonzero(np.triu(S >= tau, 1))
    for i, j in zip(ii, jj):
        if y[i] == y[j]:
            union(i, j)
    if block:
        num = df.relative_path.str.extract(r"img_(\d+)")[0].astype(int).values
        for c in np.unique(y):
            ix = np.nonzero(y == c)[0]
            ix = ix[np.argsort(num[ix])]
            for b in range(0, len(ix), block):
                for j in ix[b + 1:b + block]:
                    union(ix[b], j)
    roots = np.array([find(i) for i in range(len(df))])
    return pd.factorize(roots)[0]


def boards(Eb, df, kmax=8):
    best = None
    for k in range(2, kmax + 1):
        lab = KMeans(k, n_init=20, random_state=0).fit_predict(Eb)
        s = silhouette_score(Eb, lab)
        print(f"board k={k} silhouette {s:.3f}")
        if best is None or s > best[0]:
            best = (s, k, lab)
    s, k, lab = best
    print(f"chosen k={k}")
    print(pd.crosstab(df.variety, lab))
    return lab


if __name__ == "__main__":
    df = load_manifest("leafwidth")
    E, Eb = leaf_embeddings()
    if sys.argv[1] == "analyse":
        analyse(df, E)
    else:
        tau, block = float(sys.argv[2]), int(sys.argv[3])
        S = E @ E.T
        g = components(df, S, tau, block)
        g_dup = components(df, S, tau)  # duplicates / near-identical shots only (sensitivity analysis)
        b = boards(Eb, df)
        pd.DataFrame({"relative_path": df.relative_path, "group": g, "group_dup": g_dup, "board": b}).to_csv(
            CACHE / "groups.csv", index=False)
        sizes = pd.Series(g).value_counts()
        info = {"tau": tau, "block": block, "n_groups_dup_only": int(pd.Series(g_dup).nunique()), "n_groups": int(sizes.size), "max_group": int(sizes.max()),
                "groups_per_class": pd.Series(g).groupby(df.variety.values).nunique().reindex(CLASSES).to_dict(),
                "board_crosstab": pd.crosstab(df.variety, b).to_dict()}
        (RESULTS / "groups_info.json").write_text(json.dumps(info, indent=1, default=int))
        print(json.dumps(info, indent=1, default=int))
