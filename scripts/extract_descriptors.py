"""Botanical descriptors (164-D, blade pixels) and the 34-D whole-image baseline descriptor of the leaf blade.

python scripts/extract_descriptors.py --data-root DATA
-> <out-dir>/features/leaf_descriptors.parquet and leaf_baseline34.parquet (rows in leaf-feature order)
"""
import argparse

import numpy as np
import pandas as pd
from PIL import Image

from agrifusenet.config import add_path_args, paths_from_args
from agrifusenet.data import read_metadata, category
from agrifusenet.descriptors import botanical_descriptors, baseline_descriptor
from agrifusenet.masks import load_mask


def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__))
    paths = paths_from_args(ap.parse_args())
    df = category(read_metadata(paths), "leafwidth")
    rows, base = [], []
    for i, (rel, p) in enumerate(zip(df.relative_path, df.path)):
        a = np.asarray(Image.open(p).convert("RGB"))
        rows.append(botanical_descriptors(a, load_mask(paths.masks("colour"), rel)))
        base.append(baseline_descriptor(a))
        if i % 100 == 0:
            print(f"{i}/{len(df)}", flush=True)
    H, B = pd.DataFrame(rows), pd.DataFrame(base)
    H.index = B.index = df.relative_path.values
    H.to_parquet(paths.features / "leaf_descriptors.parquet")
    B.to_parquet(paths.features / "leaf_baseline34.parquet")
    print("botanical descriptors", H.shape, "| baseline descriptor", B.shape,
          "| groups", H.columns.str.split("_").str[0].value_counts().to_dict())


if __name__ == "__main__":
    main()
