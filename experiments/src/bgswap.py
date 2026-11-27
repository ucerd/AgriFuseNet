"""Background-swap counterfactuals: each leaf is pasted onto the (inpainted) board of another variety.

python bgswap.py make            -> cache/swap/*.jpg composites + cache/swap_meta.csv (partner index/label)
python bgswap.py feats <backbone> -> cache/swap_<backbone>.npz  (full, leaf, tiles; composites re-segmented)
python bgswap.py hand            -> cache/swap_handcrafted.parquet
"""
import sys
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from common import load_manifest, load_rgb, CACHE
from segment import load_mask, leaf_mask

SWAP = CACHE / "swap"


def composite(fg_rgb, fg_m, bg_rgb, bg_m):
    hole = cv2.dilate(bg_m.astype(np.uint8), np.ones((21, 21), np.uint8))
    bg = cv2.inpaint(bg_rgb, hole, 7, cv2.INPAINT_TELEA)
    soft = cv2.GaussianBlur(fg_m.astype(np.float32), (5, 5), 0)[..., None]
    return (soft * fg_rgb + (1 - soft) * bg).astype(np.uint8)


def make():
    SWAP.mkdir(parents=True, exist_ok=True)
    df = load_manifest("leafwidth")
    rng = np.random.default_rng(2024)
    partners = []
    for i, (rel, p) in enumerate(zip(df.relative_path, df.path)):
        j = int(rng.choice(np.nonzero(df.y.values != df.y.values[i])[0]))
        c = composite(load_rgb(p), load_mask(rel), load_rgb(df.path[j]), load_mask(df.relative_path[j]))
        Image.fromarray(c).save(SWAP / f"{i:04d}.jpg", quality=95)
        partners.append(j)
    pd.DataFrame({"partner": partners, "partner_y": df.y.values[partners]}).to_csv(CACHE / "swap_meta.csv", index=False)


def iter_composites():
    df = load_manifest("leafwidth")
    for i in range(len(df)):
        c = np.asarray(Image.open(SWAP / f"{i:04d}.jpg").convert("RGB"))
        yield c, leaf_mask(c)  # the deployed pipeline re-segments every input


@torch.no_grad()
def feats(backbone):
    from extract_deep import build, to_tensor
    from views import full_view, leaf_view, leaf_tiles
    model, fwd = build(backbone)
    out = {"full": [], "leaf": [], "tiles": []}
    for c, m in iter_composites():
        f = fwd(torch.stack([to_tensor(v) for v in [full_view(c), leaf_view(c, m)] + leaf_tiles(c, m)])).numpy()
        out["full"].append(f[0]); out["leaf"].append(f[1]); out["tiles"].append(f[2:6].mean(0))
    np.savez_compressed(CACHE / f"swap_{backbone}.npz", **{k: np.stack(v) for k, v in out.items()})


def hand():
    from handcrafted import extract, legacy_extract
    rows, leg = [], []
    for c, m in iter_composites():
        rows.append(extract(c, m)); leg.append(legacy_extract(c))
    pd.DataFrame(rows).to_parquet(CACHE / "swap_handcrafted.parquet")
    pd.DataFrame(leg).to_parquet(CACHE / "swap_handcrafted_legacy.parquet")


if __name__ == "__main__":
    torch.set_num_threads(8)
    {"make": make, "feats": lambda: feats(sys.argv[2]), "hand": hand}[sys.argv[1]]()
