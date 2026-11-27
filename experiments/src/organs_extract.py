"""Frozen DINOv2 features for every image of every organ category in the dataset.

Views per image: full frame, and a border probe (2 x 2 mosaic of the four 112-px corner patches), the cheapest
mask-free proxy for "what the background alone tells the classifier". With n_aug > 0, augmented copies of the
full frame are added for training-fold use.

Usage: python organs_extract.py [n_aug]   ->   cache/organs_dinov2_s.npz (relative_path, full, border, aug_full)
"""
import sys
import time
import numpy as np
import pandas as pd
import torch
from common import DATA, CACHE, load_rgb
from extract_deep import to_tensor, augment

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def border_view(a, c=112):
    H, W = a.shape[:2]
    p = [a[:c, :c], a[:c, W - c:], a[H - c:, :c], a[H - c:, W - c:]]
    return np.vstack([np.hstack(p[:2]), np.hstack(p[2:])])


@torch.no_grad()
def run(n_aug=0):
    import timm
    m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, img_size=224).eval().to(DEV)

    def fwd(batch):
        t = m.forward_features(torch.stack([to_tensor(v) for v in batch]).to(DEV))
        return torch.cat([t[:, 0], t[:, 1:].mean(1)], 1).cpu().numpy()

    df = pd.read_csv(DATA / "metadata" / "samples.csv").sort_values("relative_path").reset_index(drop=True)
    full, border, aug = [], [], [[] for _ in range(n_aug)]
    gen = np.random.default_rng(12345)
    t0 = time.time()
    for i, rel in enumerate(df.relative_path):
        a = load_rgb(str(DATA / rel))
        views = [a, border_view(a)]
        for _ in range(n_aug):
            views.append(augment(a, np.ones(a.shape[:2], bool), gen)[0])
        f = fwd(views)
        full.append(f[0]); border.append(f[1])
        for k in range(n_aug):
            aug[k].append(f[2 + k])
        if i % 500 == 0:
            print(i, "/", len(df), "%.0fs" % (time.time() - t0), flush=True)
    save = {"relative_path": df.relative_path.values.astype(str), "full": np.stack(full).astype(np.float32),
            "border": np.stack(border).astype(np.float32)}
    if n_aug:
        save["aug_full"] = np.stack([np.stack(x) for x in aug]).astype(np.float32)
    np.savez_compressed(CACHE / "organs_dinov2_s.npz", **save)
    print("done", {k: v.shape for k, v in save.items()}, "%.0fs" % (time.time() - t0))


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
