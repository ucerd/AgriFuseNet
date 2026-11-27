"""Data for the background-audit figure: balanced accuracy of four views of every organ image
(full image, four corners, background only, organ only), grouped and session-level CV.

Inputs : results/organs_feasibility_full5.json (views full, border), results/organs_feasibility_obj5.json (obj, bgonly)
Output : figures/data/fig_audit.dat  (one row per organ; columns <view>_<protocol> in %)
"""
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
OUT = HERE / "data"
OUT.mkdir(exist_ok=True)
ORGANS = [("leafwidth", "Leaf blade"), ("leavePattern", "Leaf top"), ("buddataset", "Bud"), ("crossSection", "Cross-section")]

full = json.loads((RES / "organs_feasibility_full5.json").read_text())["per_category"]
obj = json.loads((RES / "organs_feasibility_obj5.json").read_text())["per_category"]
rows = []
for i, (o, name) in enumerate(ORGANS):
    r = {"x": i, "organ": name.replace(" ", "~")}
    for proto, key in (("g", "grouped_bal"), ("s", "session_bal")):
        r[f"full_{proto}"] = 100 * full[o]["full"][key]
        r[f"corner_{proto}"] = 100 * full[o]["border"][key]
        r[f"bg_{proto}"] = 100 * obj[o]["bgonly"][key]
        r[f"organ_{proto}"] = 100 * obj[o]["obj"][key]
    rows.append(r)
d = pd.DataFrame(rows)
d.to_csv(OUT / "fig_audit.dat", sep=" ", index=False, float_format="%.1f")
print(d.round(1).to_string(index=False))
