"""Leakage groups, capture periods and (leaf blade) background boards of every organ category.

python scripts/build_groups.py --data-root DATA
-> <out-dir>/groups/<category>.csv with relative_path, variety, y, group, duplicate_group, period [, board]
   group           : duplicates, near-duplicates and capture-order blocks (used for grouped splits)
   duplicate_group : duplicates and near-duplicates only (sensitivity analysis)
   period          : capture period 0-2 within the variety (session-level splits)
   board           : background-board cluster of the leaf-blade images
Requires features/leaf_dinov2_s.npz (leaf blade) and features/frames_dinov2_s.npz (other organs).
"""
import argparse

import numpy as np
import pandas as pd

from agrifusenet.config import (ORGANS, GROUP_EMBEDDING, NEAR_DUPLICATE_SIMILARITY, CAPTURE_BLOCK,
                                add_path_args, paths_from_args)
from agrifusenet.data import read_metadata, category, leakage_groups, capture_periods, background_boards


def rows_of(npz, relative_paths):
    index = {p: i for i, p in enumerate(npz["relative_path"])}
    return np.array([index[p] for p in relative_paths])


def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--categories", nargs="+", default=ORGANS)
    args = ap.parse_args()
    paths = paths_from_args(args)
    meta = read_metadata(paths)
    leaf = np.load(paths.features / "leaf_dinov2_s.npz")
    frames = np.load(paths.features / "frames_dinov2_s.npz")
    for cat in args.categories:
        df = category(meta, cat)
        if GROUP_EMBEDDING[cat] == "leaf":
            r = rows_of(leaf, df.relative_path)
            E, bg = leaf["leaf"][r], leaf["background_cells"][r]
        else:
            E, bg = frames["frame"][rows_of(frames, df.relative_path)], None
        out = df[["relative_path", "variety", "y"]].copy()
        out["group"] = leakage_groups(df, E, NEAR_DUPLICATE_SIMILARITY, CAPTURE_BLOCK)
        out["duplicate_group"] = leakage_groups(df, E, NEAR_DUPLICATE_SIMILARITY, block=None)
        out["period"] = capture_periods(df)
        if bg is not None:
            out["board"] = background_boards(bg)
            print(pd.crosstab(out.variety, out.board))
        out.to_csv(paths.groups / f"{cat}.csv", index=False)
        print(f"{cat}: {len(df)} images, {out.group.nunique()} leakage groups "
              f"({out.duplicate_group.nunique()} without capture blocks)")


if __name__ == "__main__":
    main()
