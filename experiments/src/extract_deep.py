"""Extract frozen deep features for every image under every view (and optional train-time augmentations).

Usage: python extract_deep.py <backbone> [n_aug]
Writes cache/deep_<backbone>.npz with arrays  full, leaf, bg, bgpatch, tiles  (N x D, tiles = mean of 4 tiles)
and, when n_aug > 0, aug_leaf / aug_tiles / aug_full of shape (n_aug, N, D).
"""
import sys
import time
import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from common import load_manifest, load_rgb, CACHE
from segment import load_mask
from views import full_view, leaf_view, bg_view, bg_patch_view, leaf_tiles, GRAY

torch.set_num_threads(8)
MEAN, STD = torch.tensor([0.485, 0.456, 0.406])[:, None, None], torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def build(name):
    import timm, torchvision
    if name == "resnet50":  # torchvision ImageNet-1k V1 weights, the same network MATLAB's resnet50 ships
        m = torchvision.models.resnet50(weights="IMAGENET1K_V1"); m.fc = torch.nn.Identity()
        return m.eval(), lambda x: m(x)
    if name == "dinov2_s":
        m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, img_size=224).eval()
        def f(x):
            t = m.forward_features(x)  # (B, 1+P, D): CLS + patch tokens
            return torch.cat([t[:, 0], t[:, 1:].mean(1)], 1)
        return m, f
    timm_names = {"effnet_b0": "efficientnet_b0.ra_in1k", "convnext_t": "convnext_tiny.fb_in22k_ft_in1k",
                  "mobilenetv3": "mobilenetv3_large_100.ra_in1k"}
    m = timm.create_model(timm_names[name], pretrained=True, num_classes=0).eval()
    return m, lambda x: m(x)


def to_tensor(a, size=224):
    im = Image.fromarray(np.ascontiguousarray(a)).resize((size, size), Image.BICUBIC)
    return (torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255 - MEAN) / STD


# Augmentations of Table 2 in the manuscript; used for training-fold images only (see experiments.py)
GEOM = T.Compose([T.RandomRotation(15), T.RandomHorizontalFlip(0.5), T.RandomAffine(0, scale=(0.9, 1.1))])
COLOR = T.ColorJitter(brightness=0.2, saturation=0.1, hue=0.02)


def augment(a, m, gen):
    torch.manual_seed(int(gen.integers(1 << 31)))
    im = T.functional.to_image(np.ascontiguousarray(a))
    mk = T.functional.to_image(m.astype(np.uint8)[..., None] * 255)
    out = GEOM(torch.cat([im, mk], 0))  # identical geometry for image and mask
    rgb = COLOR(out[:3])
    return rgb.permute(1, 2, 0).numpy(), out[3].numpy() > 127


def views_of(a, m):
    return [full_view(a), leaf_view(a, m), bg_view(a, m), bg_patch_view(a, m)] + leaf_tiles(a, m)


@torch.no_grad()
def run(name, n_aug=0):
    model, fwd = build(name)
    df = load_manifest("leafwidth")
    out = {k: [] for k in ("full", "leaf", "bg", "bgpatch", "tiles")}
    aug = {k: [[] for _ in range(n_aug)] for k in ("full", "leaf", "tiles")}
    gen = np.random.default_rng(12345)
    t0 = time.time()
    for i, (rel, p) in enumerate(zip(df.relative_path, df.path)):
        a, m = load_rgb(p), load_mask(rel)
        batch = views_of(a, m)
        for _ in range(n_aug):
            aa, mm = augment(a, m, gen)
            if mm.sum() < 500:
                mm = m
            batch += [aa, leaf_view(aa, mm)] + leaf_tiles(aa, mm)
        f = fwd(torch.stack([to_tensor(v) for v in batch])).numpy()
        out["full"].append(f[0]); out["leaf"].append(f[1]); out["bg"].append(f[2]); out["bgpatch"].append(f[3])
        out["tiles"].append(f[4:8].mean(0))
        for k in range(n_aug):
            g = f[8 + 6 * k: 8 + 6 * (k + 1)]
            aug["full"][k].append(g[0]); aug["leaf"][k].append(g[1]); aug["tiles"][k].append(g[2:6].mean(0))
        if i % 100 == 0:
            print(name, i, "/", len(df), "%.1fs" % (time.time() - t0), flush=True)
    save = {k: np.stack(v).astype(np.float32) for k, v in out.items()}
    for k, v in aug.items():
        if n_aug:
            save["aug_" + k] = np.stack([np.stack(x) for x in v]).astype(np.float32)
    np.savez_compressed(CACHE / f"deep_{name}.npz", **save)
    print("done", name, {k: v.shape for k, v in save.items()}, "%.1fs" % (time.time() - t0))


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
