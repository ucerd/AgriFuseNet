"""Dataset metadata, curation, leakage groups, capture periods and background boards."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from .config import VARIETIES, NEAR_DUPLICATE_SIMILARITY, CAPTURE_BLOCK, N_PERIODS


def read_metadata(paths):
    """All rows of metadata/samples.csv with absolute image paths and integer labels."""
    meta = pd.read_csv(paths.metadata)
    meta["path"] = [str(Path(paths.data_root) / p) for p in meta.relative_path]
    meta["y"] = meta.variety.map(VARIETIES.index)
    return meta


def ambiguous(meta):
    """Boolean mask of files whose content (SHA-256) occurs under more than one variety."""
    return meta.groupby("hash_sha256").variety.transform("nunique") > 1


def category(meta, name, curated=True):
    """Images of one organ category, sorted by relative path. With curated=True, files whose content occurs
    under more than one variety are removed, because their label cannot be determined."""
    keep = meta.feature_type == name
    if curated:
        keep &= ~ambiguous(meta)
    return meta[keep].sort_values("relative_path").reset_index(drop=True)


def capture_index(df):
    """Capture order of every image (the running number in the file name)."""
    return df.relative_path.str.extract(r"img_(\d+)")[0].astype(int).values


def capture_periods(df, n=N_PERIODS):
    """Divide the capture sequence of every variety into n contiguous periods of equal size."""
    num, y = capture_index(df), df.y.values
    period = np.zeros(len(df), int)
    for c in np.unique(y):
        ix = np.nonzero(y == c)[0]
        ix = ix[np.argsort(num[ix])]
        for k, part in enumerate(np.array_split(ix, n)):
            period[part] = k
    return period


def leakage_groups(df, embedding, tau=NEAR_DUPLICATE_SIMILARITY, block=CAPTURE_BLOCK):
    """Union-find over (i) byte-identical files, (ii) same-variety pairs with cosine similarity >= tau and
    (iii) contiguous capture-order blocks of `block` images within each variety. Returns one integer group
    label per image; images of one group are never separated by a train/test split."""
    E = normalize(embedding)
    S = E @ E.T
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
        num = capture_index(df)
        for c in np.unique(y):
            ix = np.nonzero(y == c)[0]
            ix = ix[np.argsort(num[ix])]
            for b in range(0, len(ix), block):
                for j in ix[b + 1:b + block]:
                    union(ix[b], j)
    roots = np.array([find(i) for i in range(len(df))])
    return pd.factorize(roots)[0]


def background_boards(embedding, kmax=8, verbose=True):
    """k-means clustering of background-only embeddings, k chosen by the silhouette criterion."""
    E = normalize(embedding)
    best = None
    for k in range(2, kmax + 1):
        lab = KMeans(k, n_init=20, random_state=0).fit_predict(E)
        s = silhouette_score(E, lab)
        if verbose:
            print(f"boards: k={k} silhouette {s:.3f}")
        if best is None or s > best[0]:
            best = (s, k, lab)
    return best[2]


def load_groups(paths, name):
    """Leakage groups, capture periods (and, for the leaf blade, background boards) of one category."""
    return pd.read_csv(paths.groups / f"{name}.csv")
