"""Text-prompted organ masks with SAM 3 (Segment Anything with Concepts, facebook/sam3).

Every instance of the organ concept above the score threshold is kept (union). If nothing is detected, the
threshold is lowered once; if still nothing, the colour mask of segment.py is used, and the method is logged.

Usage: python sam3_masks.py <category> [<category> ...] [--sample N] [--prompt "text"]
Writes cache/masks_sam3/<relative_path>.png and results/sam3_<category>.csv (score, coverage, method).
"""
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from common import DATA, CACHE, RESULTS, CLASSES

# chosen by per-prompt detection rate on 6 images per variety (sam3_masks.py --try ...)
PROMPTS = {"leafwidth": ["long narrow leaf", "grass blade", "green leaf"],
           "leavePattern": ["plant", "sugarcane leaf", "grass blade"],
           "buddataset": ["sugarcane stalk"], "crossSection": ["food", "slice"],
           "internodeDistance": ["sugarcane stalk"], "individualSugarcane": ["sugarcane stalk"]}
OUT = CACHE / "masks_sam3"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_sam3():
    from transformers import Sam3Model, Sam3Processor
    model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch.bfloat16 if DEV.type == "cuda" else torch.float32).to(DEV).eval()
    return model, Sam3Processor.from_pretrained("facebook/sam3")


@torch.no_grad()
def segment(model, proc, img, prompts, thr=0.5):
    """Union of all instance masks for any of the prompts, best score, number of instances."""
    H, W = img.size[1], img.size[0]
    union, best, n = np.zeros((H, W), bool), 0.0, 0
    for p in prompts:
        inp = proc(images=img, text=p, return_tensors="pt").to(DEV)
        inp = {k: (v.to(model.dtype) if torch.is_floating_point(v) else v) for k, v in inp.items()}
        out = model(**inp)
        r = proc.post_process_instance_segmentation(out, threshold=thr, mask_threshold=0.5,
                                                    target_sizes=inp["original_sizes"].tolist())[0]
        for m, s in zip(r["masks"], r["scores"]):
            union |= m.cpu().numpy().astype(bool); best = max(best, float(s)); n += 1
    return union, best, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cats", nargs="+")
    ap.add_argument("--sample", type=int, default=0, help="only N images per variety (prompt testing)")
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--try", dest="trial", action="append", default=None,
                    help="score each candidate prompt separately on the sample (no masks written)")
    a = ap.parse_args()
    model, proc = load_sam3()
    df = pd.read_csv(DATA / "metadata" / "samples.csv").sort_values("relative_path")
    if a.trial:
        for cat in a.cats:
            sub = df[df.feature_type == cat]
            sub = pd.concat([g.iloc[np.linspace(0, len(g) - 1, a.sample or 6).astype(int)] for _, g in sub.groupby("variety")])
            imgs = [Image.open(DATA / rel).convert("RGB") for rel in sub.relative_path]
            for p in a.trial:
                res = [segment(model, proc, im, [p]) for im in imgs]
                det = np.mean([r[0].any() for r in res]); sc = np.mean([r[1] for r in res])
                cov = np.median([r[0].mean() for r in res if r[0].any()] or [0]); inst = np.mean([r[2] for r in res])
                print(f"TRY {cat:14s} {p!r:45s} detected {100*det:5.1f}% score {sc:.2f} coverage {cov:.3f} instances {inst:.1f}", flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for cat in a.cats:
        sub = df[df.feature_type == cat]
        if a.sample:
            sub = pd.concat([g.iloc[np.linspace(0, len(g) - 1, a.sample).astype(int)] for _, g in sub.groupby("variety")])
        prompts = a.prompt or PROMPTS[cat]
        rows = []
        for i, rel in enumerate(sub.relative_path):
            img = Image.open(DATA / rel).convert("RGB")
            m, s, n = segment(model, proc, img, prompts)
            method = "sam3"
            if not m.any():
                m, s, n = segment(model, proc, img, prompts, thr=0.3); method = "sam3_low"
            if not m.any():
                from segment import organ_mask, leaf_mask
                rgb = np.asarray(img)
                m = (leaf_mask(rgb) if cat == "leafwidth" else organ_mask(rgb, cat)) > 0; method = "colour_fallback"
            Image.fromarray(m.astype(np.uint8) * 255).save(OUT / (rel.replace("/", "__") + ".png"))
            rows.append({"relative_path": rel, "score": s, "instances": n, "coverage": float(m.mean()), "method": method})
            if i % 200 == 0:
                print(cat, i, "/", len(sub), flush=True)
        r = pd.DataFrame(rows)
        tag = "_sample" if a.sample else ""
        r.to_csv(RESULTS / f"sam3_{cat}{tag}.csv", index=False)
        print(cat, "prompts", prompts, "| methods", r.method.value_counts().to_dict(),
              "| coverage median %.3f" % r.coverage.median(), "| score median %.2f" % r.score.median(), flush=True)


if __name__ == "__main__":
    main()
