"""Counterfactual background swap for the leaf blade: every blade is pasted onto the board of an image of a
different variety, whose own blade is removed by inpainting. Composites are re-segmented like any new input.

python scripts/background_swap.py --data-root DATA make                      -> background_swap/images, pairs.csv
python scripts/background_swap.py --data-root DATA features --backbone B     -> features/swap_<B>.npz
python scripts/background_swap.py --data-root DATA descriptors               -> features/swap_descriptors.parquet,
                                                                                features/swap_baseline34.parquet
Evaluate with scripts/evaluate_leaf.py --test-on swapped.
"""
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from agrifusenet.backbones import BACKBONES, build, to_tensor
from agrifusenet.config import add_path_args, paths_from_args
from agrifusenet.data import read_metadata, category
from agrifusenet.descriptors import botanical_descriptors, baseline_descriptor
from agrifusenet.masks import load_mask, leaf_mask
from agrifusenet.views import masked_view, blade_tiles


def composite(fg_rgb, fg_m, bg_rgb, bg_m):
    hole = cv2.dilate(bg_m.astype(np.uint8), np.ones((21, 21), np.uint8))
    bg = cv2.inpaint(bg_rgb, hole, 7, cv2.INPAINT_TELEA)
    soft = cv2.GaussianBlur(fg_m.astype(np.float32), (5, 5), 0)[..., None]
    return (soft * fg_rgb + (1 - soft) * bg).astype(np.uint8)


def make(paths, df):
    folder = paths.swap / "images"
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2024)
    rgb = lambda i: np.asarray(Image.open(df.path[i]).convert("RGB"))
    mask = lambda i: load_mask(paths.masks("colour"), df.relative_path[i])
    partners = []
    for i in range(len(df)):
        j = int(rng.choice(np.nonzero(df.y.values != df.y.values[i])[0]))
        Image.fromarray(composite(rgb(i), mask(i), rgb(j), mask(j))).save(folder / f"{i:04d}.jpg", quality=95)
        partners.append(j)
    pd.DataFrame({"relative_path": df.relative_path, "partner": df.relative_path.values[partners],
                  "partner_y": df.y.values[partners]}).to_csv(paths.swap / "pairs.csv", index=False)


def composites(paths, n):
    for i in range(n):
        c = np.asarray(Image.open(paths.swap / "images" / f"{i:04d}.jpg").convert("RGB"))
        yield c, leaf_mask(c)


@torch.no_grad()
def features(paths, n, backbone, device):
    _, forward = build(backbone, device)
    out = {"full": [], "leaf": [], "tiles": []}
    for c, m in composites(paths, n):
        f = forward(torch.stack([to_tensor(v) for v in [c, masked_view(c, m)] + blade_tiles(c, m)])).numpy()
        out["full"].append(f[0]); out["leaf"].append(f[1]); out["tiles"].append(f[2:6].mean(0))
    np.savez_compressed(paths.features / f"swap_{backbone}.npz", **{k: np.stack(v) for k, v in out.items()})


def descriptors(paths, n):
    rows, base = [], []
    for c, m in composites(paths, n):
        rows.append(botanical_descriptors(c, m))
        base.append(baseline_descriptor(c))
    pd.DataFrame(rows).to_parquet(paths.features / "swap_descriptors.parquet")
    pd.DataFrame(base).to_parquet(paths.features / "swap_baseline34.parquet")


def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("step", choices=["make", "features", "descriptors"])
    ap.add_argument("--backbone", choices=BACKBONES, default="dinov2_s")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    paths = paths_from_args(args)
    df = category(read_metadata(paths), "leafwidth")
    if args.step == "make":
        make(paths, df)
    elif args.step == "features":
        features(paths, len(df), args.backbone, args.device)
    else:
        descriptors(paths, len(df))


if __name__ == "__main__":
    main()
