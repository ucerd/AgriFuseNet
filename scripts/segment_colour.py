"""Colour-based segmentation of one organ category (used for the leaf-blade encoder).

python scripts/segment_colour.py --data-root DATA --category leafwidth
-> <out-dir>/masks/colour/<category>/<variety>/<image>.png

GrabCut draws its initialisation from OpenCV's global random-number generator, so the masks are computed for the
whole category in a fixed order (sorted relative paths) within one process.
"""
import argparse

import numpy as np
from PIL import Image

from agrifusenet.config import add_path_args, paths_from_args
from agrifusenet.data import read_metadata, category
from agrifusenet.masks import colour_mask, save_mask


def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--category", default="leafwidth")
    args = ap.parse_args()
    paths = paths_from_args(args)
    df = category(read_metadata(paths), args.category)
    cover = []
    for i, (rel, p) in enumerate(zip(df.relative_path, df.path)):
        m = colour_mask(np.asarray(Image.open(p).convert("RGB")), args.category)
        save_mask(paths.masks("colour"), rel, m)
        cover.append(m.mean())
        if i % 200 == 0:
            print(f"{i}/{len(df)}", flush=True)
    cover = np.array(cover)
    print(f"{args.category}: {len(df)} masks, organ coverage mean {cover.mean():.3f} "
          f"(min {cover.min():.3f}, max {cover.max():.3f})")


if __name__ == "__main__":
    main()
