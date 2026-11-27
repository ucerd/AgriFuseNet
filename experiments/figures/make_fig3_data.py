"""Data files for Fig. 3 (accuracy vs. number of organs; confusion matrix of the four-organ model).

Inputs : results/organ_fusion/main_*.csv  (accuracy of every organ subset, every split)
         results/organ_fusion/main_cm_session_s0.csv  (session-level run with confusion matrices)
Outputs: figures/data/fig3_<protocol>_<method>_mean.dat   n_organs, mean accuracy over subsets
         figures/data/fig3_<protocol>_<method>_subsets.dat n_organs, accuracy of each subset
         figures/data/fig3_confusion.dat                 true, pred, row-normalised % (session, organ-set, 4 organs)
"""
import ast
import glob
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "organ_fusion"
OUT = HERE / "data"
OUT.mkdir(exist_ok=True)
ALL = "leafwidth+leavePattern+buddataset+crossSection"

df = pd.concat([pd.read_csv(f) for f in glob.glob(str(RES / "main_[gs]*.csv"))], ignore_index=True)
df = df[df.variant == "main"]
for proto in ("grouped", "session"):
    for m in ("set_mean", "late_sum"):
        d = df[(df.protocol == proto) & (df.method == m)]
        s = d.groupby(["subset", "n_organs"]).acc.mean().mul(100).reset_index()
        s[["n_organs", "acc"]].to_csv(OUT / f"fig3_{proto}_{m}_subsets.dat", sep=" ", index=False, float_format="%.2f")
        s.groupby("n_organs").acc.mean().reset_index().to_csv(OUT / f"fig3_{proto}_{m}_mean.dat", sep=" ", index=False,
                                                              float_format="%.2f")

cm_file = RES / "main_cm_session_s0.csv"
if cm_file.exists():
    c = pd.read_csv(cm_file)
    c = c[(c.method == "set_mean") & (c.subset == ALL)]
    cm = np.sum([np.array(ast.literal_eval(x)) for x in c.confusion], axis=0).astype(float)
    cm = 100 * cm / cm.sum(1, keepdims=True)
    rows = [(i, j, cm[i, j]) for i in range(5) for j in range(5)]
    pd.DataFrame(rows, columns=["true", "pred", "pct"]).to_csv(OUT / "fig3_confusion.dat", sep=" ", index=False,
                                                              float_format="%.1f")
    print("confusion (row %):\n", np.round(cm, 1), "\naccuracy of this run: %.2f" % (100 * c.acc.mean()))
print("data written to", OUT)
