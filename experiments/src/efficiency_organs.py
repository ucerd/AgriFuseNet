"""Per-stage inference cost of the multi-organ pipeline (GPU): SAM 3 organ segmentation, DINOv2 organ encoding,
organ-set fusion. Median wall time over repeated runs, parameter counts and peak GPU memory.

Usage: python efficiency_organs.py   ->  results/efficiency_organs.json
"""
import json
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from common import DATA, RESULTS
from sam3_masks import load_sam3, segment, PROMPTS
from organs_obj_extract import obj_view
from extract_deep import to_tensor
from organ_fusion import OrganSetNet

DEV = torch.device("cuda")
ORGANS = ["leafwidth", "leavePattern", "buddataset", "crossSection"]


def timed(fn, n=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1000)


def main():
    df = pd.read_csv(DATA / "metadata" / "samples.csv")
    imgs = {o: Image.open(DATA / df[df.feature_type == o].relative_path.iloc[0]).convert("RGB") for o in ORGANS}
    torch.cuda.reset_peak_memory_stats()
    model, proc = load_sam3()
    import timm
    vit = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, img_size=224).eval().to(DEV)
    res = {"gpu": torch.cuda.get_device_name(0),
           "params_M": {"sam3": sum(p.numel() for p in model.parameters()) / 1e6,
                        "dinov2_s": sum(p.numel() for p in vit.parameters()) / 1e6}}
    seg_ms, enc_ms = {}, {}
    for o in ORGANS:
        seg_ms[o] = timed(lambda: segment(model, proc, imgs[o], PROMPTS[o]), n=10)
        m, _, _ = segment(model, proc, imgs[o], PROMPTS[o])
        crop = obj_view(np.asarray(imgs[o]), m)
        x = to_tensor(crop)[None].to(DEV)
        with torch.no_grad():
            enc_ms[o] = timed(lambda: vit.forward_features(x))
    res["sam3_ms_per_image"] = seg_ms
    res["dinov2_ms_per_crop"] = enc_ms
    net = OrganSetNet([[768, 768, 164], [768], [768], [768]], pool="mean").to(DEV).eval()
    res["params_M"]["organ_set_net"] = sum(p.numel() for p in net.parameters()) / 1e6
    xs = [[torch.randn(1, 768, device=DEV), torch.randn(1, 768, device=DEV), torch.randn(1, 164, device=DEV)]] + \
         [[torch.randn(1, 768, device=DEV)] for _ in range(3)]
    present = torch.ones(1, 4, dtype=torch.bool, device=DEV)
    with torch.no_grad():
        res["fusion_ms_per_specimen"] = timed(lambda: net(xs, present), n=100)
    res["total_ms_per_specimen"] = sum(seg_ms.values()) + sum(enc_ms.values()) + res["fusion_ms_per_specimen"]
    res["peak_gpu_memory_MB"] = torch.cuda.max_memory_allocated() / 2 ** 20
    (RESULTS / "efficiency_organs.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
