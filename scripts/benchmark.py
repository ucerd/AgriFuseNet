"""Inference cost.

python scripts/benchmark.py --data-root DATA leaf --backbone dinov2_s
    end-to-end cost of the leaf-blade encoder per image on the CPU (batch size 1, 4 threads, median over 40
    images after warm-up): segmentation, views, backbone, descriptors, fusion and SVM; parameters, model size
    and peak memory; the full-image baseline (backbone and SVM) for comparison.
python scripts/benchmark.py --data-root DATA organs
    cost of the four-organ system per specimen on the GPU: SAM 3 segmentation, DINOv2 crop encoding and
    organ-set fusion; parameters and peak GPU memory.
Writes <out-dir>/results/benchmark_<mode>[_<backbone>].json
"""
import argparse
import io
import json
import pickle
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from agrifusenet.backbones import BACKBONES, build, to_tensor
from agrifusenet.config import ORGANS, SAM3_PROMPTS, add_path_args, paths_from_args
from agrifusenet.data import read_metadata, category


def nbytes(obj):
    if isinstance(obj, torch.nn.Module):
        buf = io.BytesIO()
        torch.save(obj.state_dict(), buf)
        return buf.tell()
    return len(pickle.dumps(obj))


def timed(fn, *a):
    t = time.perf_counter()
    r = fn(*a)
    return r, (time.perf_counter() - t) * 1000


def leaf(paths, backbone, n=40, warmup=5):
    import psutil
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from agrifusenet.descriptors import botanical_descriptors
    from agrifusenet.leaf_encoder import LeafEncoder
    from agrifusenet.masks import leaf_mask
    from agrifusenet.views import masked_view, blade_tiles

    torch.set_num_threads(4)
    proc = psutil.Process()
    df = category(read_metadata(paths), "leafwidth")
    y = df.y.values
    d = np.load(paths.features / f"leaf_{backbone}.npz")
    H = pd.read_parquet(paths.features / "leaf_descriptors.parquet").values
    model, forward = build(backbone, "cpu")
    # classifiers fitted on all images: the cost does not depend on which images were used
    baseline = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")).fit(d["full"], y)
    X = np.hstack([d["leaf"], d["tiles"], H]).astype(np.float32)
    s0, s1 = d["leaf"].shape[1], d["leaf"].shape[1] + d["tiles"].shape[1]
    encoder = LeafEncoder(slices=[slice(0, s0), slice(s0, s1), slice(s1, X.shape[1])], epochs=30).fit(X, y)
    idx = np.random.default_rng(0).choice(len(df), n + warmup, replace=False)
    rows = {"baseline": [], "encoder": []}
    rss0 = proc.memory_info().rss
    peak = rss0
    with torch.no_grad():
        for k, i in enumerate(idx):
            a = np.asarray(Image.open(df.path[i]).convert("RGB"))
            t = {}
            x, t["preprocess"] = timed(lambda: to_tensor(a)[None])
            f, t["backbone"] = timed(lambda: forward(x).numpy())
            _, t["classifier"] = timed(baseline.predict, f)
            t["total"] = sum(t.values())
            u = {}
            m, u["segmentation"] = timed(leaf_mask, a)
            xb, u["views"] = timed(lambda: torch.stack([to_tensor(v) for v in [masked_view(a, m)] + blade_tiles(a, m)]))
            fb, u["backbone"] = timed(lambda: forward(xb).numpy())
            h, u["descriptors"] = timed(lambda: np.array(list(botanical_descriptors(a, m).values()), np.float32))
            z = np.hstack([fb[0], fb[1:].mean(0), h])[None]
            _, u["fusion_svm"] = timed(encoder.predict, z)
            u["total"] = sum(u.values())
            peak = max(peak, proc.memory_info().rss)
            if k >= warmup:
                rows["baseline"].append(t)
                rows["encoder"].append(u)
    return {"backbone": backbone, "median_ms": {k: pd.DataFrame(v).median().round(2).to_dict() for k, v in rows.items()},
            "backbone_params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
            "backbone_MB": round(nbytes(model) / 2 ** 20, 1),
            "fusion_params_K": round(sum(p.numel() for p in encoder.net_.parameters()) / 1e3, 1),
            "fusion_MB": round(nbytes(encoder.net_) / 2 ** 20, 2),
            "svm_baseline_MB": round(nbytes(baseline) / 2 ** 20, 2), "svm_encoder_MB": round(nbytes(encoder.svm_) / 2 ** 20, 2),
            "peak_memory_increase_MB": round((peak - rss0) / 2 ** 20, 1),
            "process_peak_MB": round(getattr(proc.memory_info(), "peak_wset", peak) / 2 ** 20, 0)}


def organs(paths):
    import timm
    from agrifusenet.organ_fusion import OrganSetNet
    from agrifusenet.sam3 import load_model, segment
    from agrifusenet.views import organ_crop

    dev = torch.device("cuda")

    def gpu_ms(fn, n=20, warm=3):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
        return float(np.median(ts) * 1000)

    meta = read_metadata(paths)
    images = {o: Image.open(category(meta, o).path.iloc[0]).convert("RGB") for o in ORGANS}
    torch.cuda.reset_peak_memory_stats()
    model, processor = load_model(dev)
    vit = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, img_size=224).eval().to(dev)
    res = {"gpu": torch.cuda.get_device_name(0),
           "params_M": {"sam3": sum(p.numel() for p in model.parameters()) / 1e6,
                        "dinov2_s": sum(p.numel() for p in vit.parameters()) / 1e6}}
    seg, enc = {}, {}
    for o in ORGANS:
        seg[o] = gpu_ms(lambda: segment(model, processor, images[o], SAM3_PROMPTS[o]), n=10)
        m = segment(model, processor, images[o], SAM3_PROMPTS[o])[0]
        x = to_tensor(organ_crop(np.asarray(images[o]), m))[None].to(dev)
        with torch.no_grad():
            enc[o] = gpu_ms(lambda: vit.forward_features(x))
    net = OrganSetNet([[768, 768, 164], [768], [768], [768]], pool="mean").to(dev).eval()
    xs = [[torch.randn(1, 768, device=dev), torch.randn(1, 768, device=dev), torch.randn(1, 164, device=dev)]] + \
         [[torch.randn(1, 768, device=dev)] for _ in range(3)]
    present = torch.ones(1, 4, dtype=torch.bool, device=dev)
    with torch.no_grad():
        res["fusion_ms_per_specimen"] = gpu_ms(lambda: net(xs, present), n=100)
    res.update(sam3_ms_per_image=seg, dinov2_ms_per_crop=enc,
               total_ms_per_specimen=sum(seg.values()) + sum(enc.values()) + res["fusion_ms_per_specimen"],
               peak_gpu_memory_MB=torch.cuda.max_memory_allocated() / 2 ** 20)
    res["params_M"]["organ_set_net"] = sum(p.numel() for p in net.parameters()) / 1e6
    return res


def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("mode", choices=["leaf", "organs"])
    ap.add_argument("--backbone", choices=BACKBONES, default="dinov2_s")
    args = ap.parse_args()
    paths = paths_from_args(args)
    res = leaf(paths, args.backbone) if args.mode == "leaf" else organs(paths)
    name = f"benchmark_leaf_{args.backbone}.json" if args.mode == "leaf" else "benchmark_organs.json"
    (paths.results / name).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
