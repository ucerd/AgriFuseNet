"""Organ-only DINOv2 features from the SAM 3 masks (cache/masks_sam3).

Views per image:
  obj    : square crop around the organ mask (10 % margin), pixels outside the mask set to the neutral grey, so
           small organs (cross-sections cover ~4 % of the frame) are seen at full resolution and no background
           pixel reaches the network;
  bgonly : the frame with the dilated organ mask greyed out (background probe for the shortcut audit);
  aug_obj: n_aug augmented copies of obj (geometry + colour jitter of the frame and mask before cropping).

Usage: python organs_obj_extract.py [n_aug]  ->  cache/organs_obj_dinov2_s.npz
"""
import sys
import time
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from common import DATA, CACHE, load_rgb
from extract_deep import to_tensor, augment
from views import GRAY

ORGANS = ["leafwidth", "leavePattern", "buddataset", "crossSection"]
MASKS = CACHE / "masks_sam3"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_sam3_mask(rel):
    return np.asarray(Image.open(MASKS / (rel.replace("/", "__") + ".png"))) > 0


def obj_view(a, m, margin=0.10):
    if not m.any():
        return np.ascontiguousarray(np.where(m[..., None], a, GRAY).astype(np.uint8))
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(max(y1 - y0, x1 - x0) * (1 + 2 * margin))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    H, W = m.shape
    canvas = np.tile(GRAY, (side, side, 1)).astype(np.uint8)
    fg = np.where(m[..., None], a, GRAY).astype(np.uint8)
    sy0, sx0 = cy - side // 2, cx - side // 2
    ya, xa = max(sy0, 0), max(sx0, 0)
    yb, xb = min(sy0 + side, H), min(sx0 + side, W)
    canvas[ya - sy0:yb - sy0, xa - sx0:xb - sx0] = fg[ya:yb, xa:xb]
    return canvas


def bg_only(a, m):
    md = cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    return np.where(md[..., None], GRAY, a).astype(np.uint8)


@torch.no_grad()
def run(n_aug=0):
    import timm
    net = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, img_size=224).eval().to(DEV)

    def fwd(batch):
        t = net.forward_features(torch.stack([to_tensor(v) for v in batch]).to(DEV))
        return torch.cat([t[:, 0], t[:, 1:].mean(1)], 1).cpu().numpy()

    df = pd.read_csv(DATA / "metadata" / "samples.csv")
    df = df[df.feature_type.isin(ORGANS)].sort_values("relative_path").reset_index(drop=True)
    obj, bgo, cover, aug = [], [], [], [[] for _ in range(n_aug)]
    gen = np.random.default_rng(12345)
    t0 = time.time()
    for i, rel in enumerate(df.relative_path):
        a, m = load_rgb(str(DATA / rel)), load_sam3_mask(rel)
        views = [obj_view(a, m), bg_only(a, m)]
        for _ in range(n_aug):
            aa, mm = augment(a, m, gen)
            views.append(obj_view(aa, mm if mm.sum() >= 100 else m))
        f = fwd(views)
        obj.append(f[0]); bgo.append(f[1]); cover.append(m.mean())
        for k in range(n_aug):
            aug[k].append(f[2 + k])
        if i % 500 == 0:
            print(i, "/", len(df), "%.0fs" % (time.time() - t0), flush=True)
    save = {"relative_path": df.relative_path.values.astype(str), "obj": np.stack(obj).astype(np.float32),
            "bgonly": np.stack(bgo).astype(np.float32), "cover": np.array(cover, np.float32)}
    if n_aug:
        save["aug_obj"] = np.stack([np.stack(x) for x in aug]).astype(np.float32)
    np.savez_compressed(CACHE / "organs_obj_dinov2_s.npz", **save)
    print("done", {k: v.shape for k, v in save.items()}, "%.0fs" % (time.time() - t0))


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
