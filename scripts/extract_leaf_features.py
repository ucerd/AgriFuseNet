"""Frozen backbone features of every leaf-blade image under every view, plus augmented training copies.

python scripts/extract_leaf_features.py --data-root DATA --backbone dinov2_s
-> <out-dir>/features/leaf_<backbone>.npz with
   full, leaf, background, background_cells, tiles (N, D; tiles = mean over 4 blade tiles) and
   aug_full, aug_leaf, aug_tiles (n_aug, N, D). Requires the colour masks of the leaf blade.
"""
import argparse
import time

import numpy as np
import torch
from PIL import Image

from agrifusenet.backbones import BACKBONES, build, to_tensor, augment
from agrifusenet.config import N_AUG_LEAF, add_path_args, paths_from_args
from agrifusenet.data import read_metadata, category
from agrifusenet.masks import load_mask
from agrifusenet.views import masked_view, background_view, background_cells, blade_tiles


@torch.no_grad()
def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--backbone", choices=BACKBONES, default="dinov2_s")
    ap.add_argument("--n-aug", type=int, default=N_AUG_LEAF)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    paths = paths_from_args(args)
    _, forward = build(args.backbone, args.device)
    df = category(read_metadata(paths), "leafwidth")
    out = {k: [] for k in ("full", "leaf", "background", "background_cells", "tiles")}
    aug = {k: [[] for _ in range(args.n_aug)] for k in ("full", "leaf", "tiles")}
    gen = np.random.default_rng(12345)
    t0 = time.time()
    for i, (rel, p) in enumerate(zip(df.relative_path, df.path)):
        a = np.asarray(Image.open(p).convert("RGB"))
        m = load_mask(paths.masks("colour"), rel)
        batch = [a, masked_view(a, m), background_view(a, m), background_cells(a, m)] + blade_tiles(a, m)
        for _ in range(args.n_aug):
            aa, mm = augment(a, m, gen)
            if mm.sum() < 500:  # the transformed mask is unusable when the blade leaves the frame
                mm = m
            batch += [aa, masked_view(aa, mm)] + blade_tiles(aa, mm)
        f = forward(torch.stack([to_tensor(v) for v in batch])).numpy()
        out["full"].append(f[0]); out["leaf"].append(f[1])
        out["background"].append(f[2]); out["background_cells"].append(f[3])
        out["tiles"].append(f[4:8].mean(0))
        for k in range(args.n_aug):
            g = f[8 + 6 * k: 8 + 6 * (k + 1)]
            aug["full"][k].append(g[0]); aug["leaf"][k].append(g[1]); aug["tiles"][k].append(g[2:6].mean(0))
        if i % 100 == 0:
            print(f"{args.backbone} {i}/{len(df)} {time.time() - t0:.0f}s", flush=True)
    save = {k: np.stack(v).astype(np.float32) for k, v in out.items()}
    if args.n_aug:
        save.update({"aug_" + k: np.stack([np.stack(x) for x in v]).astype(np.float32) for k, v in aug.items()})
    save["relative_path"] = df.relative_path.values.astype(str)
    np.savez_compressed(paths.features / f"leaf_{args.backbone}.npz", **save)
    print("saved", {k: v.shape for k, v in save.items()})


if __name__ == "__main__":
    main()
