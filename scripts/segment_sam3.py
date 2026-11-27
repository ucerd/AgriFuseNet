"""Organ masks with Segment Anything 3 and concept prompts.

Masks:   python scripts/segment_sam3.py --data-root DATA --categories leafwidth leavePattern buddataset crossSection
         -> <out-dir>/masks/sam3/<category>/<variety>/<image>.png and results/sam3_<category>.csv
Prompts: python scripts/segment_sam3.py --data-root DATA --categories buddataset --rank-prompts "sugarcane stalk" "bud"
         ranks candidate prompts by detection rate and mean score on a calibration subset (6 images per variety).
"""
import argparse

import numpy as np
import pandas as pd
from PIL import Image

from agrifusenet.config import ORGANS, SAM3_PROMPTS, add_path_args, paths_from_args
from agrifusenet.data import read_metadata, category
from agrifusenet.masks import save_mask
from agrifusenet.sam3 import load_model, segment, segment_organ


def calibration_subset(df, n):
    return pd.concat([g.iloc[np.linspace(0, len(g) - 1, n).astype(int)] for _, g in df.groupby("variety")])


def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--categories", nargs="+", default=ORGANS)
    ap.add_argument("--rank-prompts", nargs="+", default=None, metavar="PROMPT")
    ap.add_argument("--calibration-size", type=int, default=6, help="images per variety for prompt ranking")
    args = ap.parse_args()
    paths = paths_from_args(args)
    meta = read_metadata(paths)
    model, processor = load_model()

    if args.rank_prompts:
        for cat in args.categories:
            sub = calibration_subset(category(meta, cat, curated=False), args.calibration_size)
            images = [Image.open(p).convert("RGB") for p in sub.path]
            rows = []
            for prompt in args.rank_prompts:
                r = [segment(model, processor, im, [prompt]) for im in images]
                rows.append({"category": cat, "prompt": prompt, "detection_rate": np.mean([x[0].any() for x in r]),
                             "mean_score": np.mean([x[1] for x in r]), "mean_instances": np.mean([x[2] for x in r])})
            print(pd.DataFrame(rows).sort_values(["detection_rate", "mean_score"], ascending=False).to_string(index=False))
        return

    for cat in args.categories:
        df = category(meta, cat, curated=False)
        rows = []
        for i, (rel, p) in enumerate(zip(df.relative_path, df.path)):
            m, score, n, source = segment_organ(model, processor, Image.open(p).convert("RGB"), cat)
            save_mask(paths.masks("sam3"), rel, m)
            rows.append({"relative_path": rel, "score": score, "instances": n, "coverage": float(m.mean()),
                         "source": source})
            if i % 200 == 0:
                print(cat, f"{i}/{len(df)}", flush=True)
        r = pd.DataFrame(rows)
        r.to_csv(paths.results / f"sam3_{cat}.csv", index=False)
        print(cat, "prompts", SAM3_PROMPTS[cat], "| mask source", r.source.value_counts().to_dict(),
              f"| median coverage {r.coverage.median():.3f} | median score {r.score.median():.2f}", flush=True)


if __name__ == "__main__":
    main()
