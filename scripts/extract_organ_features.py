"""Frozen DINOv2 ViT-S/14 features of the organ images.

python scripts/extract_organ_features.py --data-root DATA frames
    every image of the dataset: full frame and corner mosaic, plus augmented copies of the frame
    -> <out-dir>/features/frames_dinov2_s.npz (relative_path, frame, corners, aug_frame)
python scripts/extract_organ_features.py --data-root DATA crops
    the four organ categories: SAM 3 organ crop (grey outside the mask) and background-only frame, plus
    augmented copies of the crop (image and mask transformed together before cropping)
    -> <out-dir>/features/organ_crops_dinov2_s.npz (relative_path, crop, background, coverage, aug_crop)
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from agrifusenet.backbones import build, to_tensor, augment
from agrifusenet.config import ORGANS, N_AUG_ORGAN, add_path_args, paths_from_args
from agrifusenet.data import read_metadata
from agrifusenet.masks import load_mask
from agrifusenet.views import corner_mosaic, organ_crop, background_view


@torch.no_grad()
def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("kind", choices=["frames", "crops"])
    ap.add_argument("--n-aug", type=int, default=N_AUG_ORGAN)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    paths = paths_from_args(args)
    _, forward = build("dinov2_s", args.device)
    meta = read_metadata(paths)
    if args.kind == "crops":
        meta = meta[meta.feature_type.isin(ORGANS)]
    df = meta.sort_values("relative_path").reset_index(drop=True)
    keys = ("frame", "corners") if args.kind == "frames" else ("crop", "background")
    out, aug, cover = {k: [] for k in keys}, [[] for _ in range(args.n_aug)], []
    gen = np.random.default_rng(12345)
    t0 = time.time()
    for i, (rel, p) in enumerate(zip(df.relative_path, df.path)):
        a = np.asarray(Image.open(p).convert("RGB"))
        if args.kind == "frames":
            views = [a, corner_mosaic(a)]
            views += [augment(a, np.ones(a.shape[:2], bool), gen)[0] for _ in range(args.n_aug)]
        else:
            m = load_mask(paths.masks("sam3"), rel)
            cover.append(m.mean())
            views = [organ_crop(a, m), background_view(a, m)]
            for _ in range(args.n_aug):
                aa, mm = augment(a, m, gen)
                views.append(organ_crop(aa, mm if mm.sum() >= 100 else m))
        f = forward(torch.stack([to_tensor(v) for v in views])).numpy()
        out[keys[0]].append(f[0]); out[keys[1]].append(f[1])
        for k in range(args.n_aug):
            aug[k].append(f[2 + k])
        if i % 500 == 0:
            print(f"{i}/{len(df)} {time.time() - t0:.0f}s", flush=True)
    save = {k: np.stack(v).astype(np.float32) for k, v in out.items()}
    save["relative_path"] = df.relative_path.values.astype(str)
    if args.n_aug:
        save["aug_" + keys[0]] = np.stack([np.stack(x) for x in aug]).astype(np.float32)
    if cover:
        save["coverage"] = np.array(cover, np.float32)
    name = "frames_dinov2_s.npz" if args.kind == "frames" else "organ_crops_dinov2_s.npz"
    np.savez_compressed(paths.features / name, **save)
    print("saved", name, {k: v.shape for k, v in save.items()})


if __name__ == "__main__":
    main()
