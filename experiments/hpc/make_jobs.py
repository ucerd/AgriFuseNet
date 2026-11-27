"""Write hpc/jobs.txt: one '<experiment-set> <name-regex>' line per remaining configuration (SLURM array task)."""
import re
from pathlib import Path

B = "dinov2_s"
names = [("protocols", f"proto_groupdup_{B}_proposed"), ("ablation", f"abl_{B}_reference")]
names += [("ablation", f"abl_{B}_proposed_{v}") for v in ("noaug", "softmaxhead", "nomodalitydrop", "k32", "k64", "k256")]
for br in ("leaf+hand", "tiles+hand", "leaf+tiles"):
    names += [("ablation", f"abl_{B}_proposed_{br}"), ("ablation", f"abl_{B}_scalar_{br}")]
names += [("ablation", f"abl_{B}_proposed_minus_{g}") for g in ("col", "glcm", "lbp", "gab", "ven", "mor")]
names += [("classifiers", f"clf_{B}_{c}") for c in ("knn", "rf", "dt", "nb")]
names += [("dedup", f"dedup_{B}_{m}") for m in ("deep_full", "proposed")]
out = Path(__file__).with_name("jobs.txt")
out.write_text("".join(f"{s} {re.escape(n)}\n" for s, n in names), newline="\n")
print(len(names), "jobs ->", out)
