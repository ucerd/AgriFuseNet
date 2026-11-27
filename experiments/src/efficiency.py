"""End-to-end inference cost on CPU (batch size 1): latency per stage, parameters, model size, peak memory.

python efficiency.py <backbone> [<backbone> ...]   -> results/efficiency.json (merged per backbone)
Timings are medians over N test images after warm-up, with torch threads fixed to 4.
"""
import sys
import io
import gc
import json
import time
import pickle
import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from common import load_manifest, load_rgb, CACHE, RESULTS
from segment import leaf_mask
from views import leaf_view, leaf_tiles
from extract_deep import build, to_tensor
from handcrafted import extract, legacy_extract
from fusion import FusionSVM

torch.set_num_threads(4)
N = 40
proc = psutil.Process()


def nbytes(obj):
    if isinstance(obj, torch.nn.Module):
        buf = io.BytesIO(); torch.save(obj.state_dict(), buf); return buf.tell()
    return len(pickle.dumps(obj))


def timed(fn, *a):
    t = time.perf_counter(); r = fn(*a); return r, (time.perf_counter() - t) * 1000


def bench(backbone):
    df = load_manifest("leafwidth"); y = df.y.values
    d = np.load(CACHE / f"deep_{backbone}.npz")
    H = pd.read_parquet(CACHE / "handcrafted.parquet").values
    L = pd.read_parquet(CACHE / "handcrafted_legacy.parquet").values
    model, fwd = build(backbone)
    n_params = sum(p.numel() for p in model.parameters())
    # classifiers fitted on all images (cost does not depend on which images were used)
    base_svm = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")).fit(d["full"], y)
    Xp = np.hstack([d["leaf"], d["tiles"], H]).astype(np.float32)
    s0, s1 = d["leaf"].shape[1], d["leaf"].shape[1] + d["tiles"].shape[1]
    prop = FusionSVM(slices=[slice(0, s0), slice(s0, s1), slice(s1, Xp.shape[1])], epochs=30).fit(Xp, y)
    fusion_params = sum(p.numel() for p in prop.net_.parameters())

    rng = np.random.default_rng(0)
    idx = rng.choice(len(df), N + 5, replace=False)
    rows = {"baseline": [], "proposed": []}
    gc.collect(); rss0 = proc.memory_info().rss
    peak = rss0
    with torch.no_grad():
        for k, i in enumerate(idx):
            a = load_rgb(df.path[i])
            # ---- baseline: full image -> backbone -> SVM (the original deep-only pipeline)
            t = {}
            x, t["preprocess"] = timed(lambda: to_tensor(a)[None])
            f, t["backbone"] = timed(lambda: fwd(x).numpy())
            _, t["classifier"] = timed(base_svm.predict, f)
            t["total"] = sum(t.values())
            # ---- proposed: segment -> leaf view + 4 rectified tiles -> backbone (one batch of 5)
            #      + leaf-only handcrafted -> attention fusion -> SVM
            u = {}
            m, u["segmentation"] = timed(leaf_mask, a)
            xb, u["views"] = timed(lambda: torch.stack([to_tensor(v) for v in [leaf_view(a, m)] + leaf_tiles(a, m)]))
            fb, u["backbone"] = timed(lambda: fwd(xb).numpy())
            h, u["handcrafted"] = timed(lambda: np.array(list(extract(a, m).values()), np.float32))
            z = np.hstack([fb[0], fb[1:].mean(0), h])[None]
            _, u["fusion+svm"] = timed(prop.predict, z)
            u["total"] = sum(u.values())
            peak = max(peak, proc.memory_info().rss)
            if k >= 5:  # warm-up
                rows["baseline"].append(t); rows["proposed"].append(u)
    _, legacy_ms = timed(lambda: [legacy_extract(load_rgb(df.path[i])) for i in idx[:10]])
    med = {k: pd.DataFrame(v).median().round(2).to_dict() for k, v in rows.items()}
    return {"backbone": backbone, "median_ms": med, "legacy_handcrafted_ms": round(legacy_ms / 10, 2),
            "backbone_params_M": round(n_params / 1e6, 2), "backbone_MB": round(nbytes(model) / 2**20, 1),
            "fusion_params_K": round(fusion_params / 1e3, 1), "fusion_MB": round(nbytes(prop.net_) / 2**20, 2),
            "svm_baseline_MB": round(nbytes(base_svm) / 2**20, 2), "svm_proposed_MB": round(nbytes(prop.svm_) / 2**20, 2),
            "n_sv_baseline": int(base_svm[-1].n_support_.sum()), "n_sv_proposed": int(prop.svm_.n_support_.sum()),
            "feature_dim_baseline": int(d["full"].shape[1]), "feature_dim_proposed_in": int(Xp.shape[1]),
            "feature_dim_proposed_fused": 128, "peak_rss_increase_MB": round((peak - rss0) / 2**20, 1),
            # whole-process peak working set (Windows); run one backbone per process for a clean figure
            "process_peak_MB": round(getattr(proc.memory_info(), "peak_wset", peak) / 2**20, 0),
            "cpu": "Intel Core i7-10610U (4C/8T), 16 GB RAM, torch threads=4, batch size 1"}


if __name__ == "__main__":
    out_path = RESULTS / "efficiency.json"
    allres = json.loads(out_path.read_text()) if out_path.exists() else {}
    for b in sys.argv[1:]:
        allres[b] = bench(b)
        print(json.dumps(allres[b], indent=1), flush=True)
        out_path.write_text(json.dumps(allres, indent=1))
