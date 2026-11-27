"""LaTeX tables and number macros for the multi-organ part of the study.

Inputs : dataset manifest, results/organs_feasibility_{full5,obj5}.json, results/organ_fusion/*.csv,
         results/sam3_*.csv, results/mask_agreement_leafwidth.csv, results/efficiency_organs.json
Outputs: revision/tables/tab_integrity.tex, tab_organ_audit.tex, tab_fusion_main.tex, tab_fusion_subsets.tex,
         tab_fusion_ablation.tex, tab_eff_organs.tex, numbers_organs.tex
"""
import json
import glob
import numpy as np
import pandas as pd
from scipy import stats
from common import DATA, CACHE, RESULTS, REV, CLASSES

TAB = REV / "tables"
ORGANS = ["leafwidth", "leavePattern", "buddataset", "crossSection"]
NAME = {"leafwidth": "Leaf blade", "leavePattern": "Leaf top", "buddataset": "Bud", "crossSection": "Cross-section",
        "internodeDistance": "Internode", "individualSugarcane": "Whole stalk", "bulk_Sugarcane_images": "Field (bulk)",
        "bulk_with_near_distance": "Field (near)"}
SHORT = {"leafwidth": "B", "leavePattern": "T", "buddataset": "U", "crossSection": "C"}
METHOD = {"late_sum": "Late fusion (score sum)", "early_concat": "Early fusion (concatenation + SVM)",
          "set_attn": "Organ-set fusion, attention pooling", "set_mean": "Organ-set fusion, mean pooling (proposed)"}
macros = {}


def mac(name, value, fmt="{:.2f}"):
    macros[name] = fmt.format(value) if not isinstance(value, str) else value


def pm(v, sd):
    return f"{100*v:.2f} $\\pm$ {100*sd:.2f}"


def corrected_t(a, b, ratio=0.25):
    """Nadeau--Bengio corrected resampled t-test for paired per-split scores (ratio = n_test / n_train)."""
    d = np.asarray(a) - np.asarray(b); n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return float("nan")
    t = d.mean() / np.sqrt((1 / n + ratio) * d.var(ddof=1))
    return float(2 * stats.t.sf(abs(t), n - 1))


def ptxt(p):
    return "--" if np.isnan(p) else ("$<$0.001" if p < 0.001 else f"{p:.3f}")


# --------------------------------------------------------------------------------------- dataset integrity
def integrity():
    m = pd.read_csv(DATA / "metadata" / "samples.csv")
    conflict = m.groupby("hash_sha256").variety.transform("nunique") > 1
    dup_within = m.duplicated(["feature_type", "variety", "hash_sha256"], keep="first")
    copied = (m.feature_type == "internodeDistance") & m.variety.isin(["CPF-251", "CPF-253"])
    rows = []
    for cat in ["leafwidth", "leavePattern", "buddataset", "crossSection", "internodeDistance", "individualSugarcane",
                "bulk_Sugarcane_images", "bulk_with_near_distance"]:
        s = m.feature_type == cat
        counts = [int((s & (m.variety == v)).sum()) for v in CLASSES]
        used = int((s & ~conflict).sum()) if cat in ORGANS else "--"
        rows.append([NAME[cat]] + counts + [int(s.sum()), used])
    lines = [r"\begin{table*}[tbp]", r"\centering", r"\footnotesize",
             r"\caption{Image categories of Sugarcane-PK v1.0 and images used in this study.}",
             r"\label{tab:integrity}", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             "Category & " + " & ".join(CLASSES) + r" & Total & Used \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join(str(x) for x in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (TAB / "tab_integrity.tex").write_text("\n".join(lines))
    mac("NAllImages", f"{len(m):,}".replace(",", "{,}"), "{}")
    mac("NConflict", str(int(conflict.sum())), "{}")
    mac("NConflictInternode", str(int((conflict & (m.feature_type == "internodeDistance")).sum())), "{}")
    used = m[m.feature_type.isin(ORGANS) & ~conflict]
    mac("NOrganImages", f"{len(used):,}".replace(",", "{,}"), "{}")


# --------------------------------------------------------------------------------------- per-organ audit
def organ_audit():
    f = json.loads((RESULTS / "organs_feasibility_full5.json").read_text())["per_category"]
    o = json.loads((RESULTS / "organs_feasibility_obj5.json").read_text())["per_category"]
    lines = [r"\begin{table*}[tbp]", r"\centering", r"\footnotesize",
             r"\caption{Balanced accuracy (\%) of four views of each organ image.}",
             r"\label{tab:organ_audit}", r"\begin{adjustbox}{max width=\textwidth}", r"\begin{tabular}{lcccccccc}", r"\toprule",
             r" & \multicolumn{4}{c}{Grouped CV} & \multicolumn{4}{c}{Session-level CV} \\ \cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             r"Organ & Full image & Corners & Background & Organ & Full image & Corners & Background & Organ \\", r"\midrule"]
    for cat in ORGANS:
        a = [f[cat]["full"]["grouped_bal"], f[cat]["border"]["grouped_bal"], o[cat]["bgonly"]["grouped_bal"], o[cat]["obj"]["grouped_bal"],
             f[cat]["full"]["session_bal"], f[cat]["border"]["session_bal"], o[cat]["bgonly"]["session_bal"], o[cat]["obj"]["session_bal"]]
        lines.append(NAME[cat] + " & " + " & ".join(f"{100*x:.1f}" for x in a) + r" \\")
        key = SHORT[cat]
        mac(f"Aud{key}Full", 100 * a[0], "{:.1f}"); mac(f"Aud{key}Corner", 100 * a[1], "{:.1f}")
        mac(f"Aud{key}Bg", 100 * a[2], "{:.1f}"); mac(f"Aud{key}Obj", 100 * a[3], "{:.1f}")
        mac(f"Aud{key}ObjSess", 100 * a[7], "{:.1f}"); mac(f"Aud{key}BgSess", 100 * a[6], "{:.1f}")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table*}"]
    (TAB / "tab_organ_audit.tex").write_text("\n".join(lines))


# --------------------------------------------------------------------------------------- fusion
def load_fusion():
    fs = glob.glob(str(RESULTS / "organ_fusion" / "*.csv"))
    return pd.concat([pd.read_csv(x) for x in fs], ignore_index=True)


SUB_ALL = "+".join(ORGANS)
SUB_LEAF = "leafwidth+leavePattern"


def fusion_main(df):
    """Method rows of the combined fusion table (written by fusion_ablation) and the text macros."""
    d = df[(df.variant == "main")]
    methods = ["late_sum", "early_concat", "set_attn", "set_mean"]
    rows = []
    ref = d[(d.method == "set_mean") & (d.protocol == "grouped") & (d.subset == SUB_ALL)].sort_values(["seed", "fold"]).acc.values
    for mth in methods:
        cells = []
        for proto in ("grouped", "session"):
            for sub in (SUB_LEAF, SUB_ALL):
                v = d[(d.method == mth) & (d.protocol == proto) & (d.subset == sub)].acc
                cells.append(pm(v.mean(), v.std(ddof=1)) if len(v) else "--")
        oth = d[(d.method == mth) & (d.protocol == "grouped") & (d.subset == SUB_ALL)].sort_values(["seed", "fold"]).acc.values
        p = corrected_t(ref, oth) if mth != "set_mean" and len(oth) == len(ref) else float("nan")
        name = METHOD[mth] if mth != "set_mean" else r"\textbf{" + METHOD[mth] + "}"
        rows.append(name + " & " + " & ".join(cells) + f" & {ptxt(p)}" + r" \\")
    # macros for the text
    for proto, P in (("grouped", "G"), ("session", "S")):
        for sub, S in ((SUB_ALL, "All"), (SUB_LEAF, "Leaf"), ("leafwidth", "Blade")):
            for mth, M in (("set_mean", "Set"), ("late_sum", "Late"), ("early_concat", "Early"), ("set_attn", "Attn")):
                v = d[(d.method == mth) & (d.protocol == proto) & (d.subset == sub)].acc
                if len(v):
                    mac(f"Fu{M}{S}{P}", 100 * v.mean()); mac(f"Fu{M}{S}{P}SD", 100 * v.std(ddof=1))
    for sub, S in ((SUB_ALL, "All"), (SUB_LEAF, "Leaf")):
        a = d[(d.method == "set_mean") & (d.protocol == "grouped") & (d.subset == sub)].sort_values(["seed", "fold"]).acc.values
        for mth, M in (("late_sum", "Late"), ("set_attn", "Attn"), ("early_concat", "Early")):
            b = d[(d.method == mth) & (d.protocol == "grouped") & (d.subset == sub)].sort_values(["seed", "fold"]).acc.values
            if len(b) == len(a) and len(a):
                mac(f"PSetVs{M}{S}", ptxt(corrected_t(a, b)), "{}")
                mac(f"WinsSetVs{M}{S}", f"{int((a > b).sum())}/{len(a)}", "{}")
    return rows


def fusion_subsets(df):
    d = df[(df.variant == "main") & df.method.isin(["late_sum", "set_mean"])]
    g = d.groupby(["protocol", "subset", "method"]).acc.mean().unstack(["protocol", "method"]) * 100
    order = sorted(g.index, key=lambda s: (s.count("+"), [ORGANS.index(x) for x in s.split("+")]))
    lines = [r"\begin{table}[tbp]", r"\centering", r"\footnotesize",
             r"\caption{Accuracy (\%) for every combination of organs (B: leaf blade, T: leaf top, U: bud, C: cross-section).}",
             r"\label{tab:fusion_subsets}", r"\begin{tabular}{lcccc}", r"\toprule",
             r" & \multicolumn{2}{c}{Grouped} & \multicolumn{2}{c}{Session} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Organs & Late & Set & Late & Set \\", r"\midrule"]
    prev = 1
    for s in order:
        k = s.count("+") + 1
        if k != prev:
            lines.append(r"\midrule"); prev = k
        lab = "+".join(SHORT[x] for x in s.split("+"))
        v = [g.loc[s].get(("grouped", "late_sum"), np.nan), g.loc[s].get(("grouped", "set_mean"), np.nan),
             g.loc[s].get(("session", "late_sum"), np.nan), g.loc[s].get(("session", "set_mean"), np.nan)]
        lines.append(lab + " & " + " & ".join(f"{x:.1f}" for x in v) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "tab_fusion_subsets.tex").write_text("\n".join(lines))
    singles = [g.loc[o][("session", "set_mean")] for o in ORGANS]
    mac("BestSingleSess", max(singles)); mac("WorstSingleSess", min(singles))


def fusion_ablation(df, method_rows):
    """Combined table: fusion methods (method_rows from fusion_main) followed by the ablation of the organ-set model."""
    rows = [("main", "set_attn", None),  # macros only; the row is in the method block
            ("abl_nodrop", "set_mean", r"\quad without organ dropout"),
            ("abl_noaug", "set_mean", r"\quad without train-time augmentation"),
            ("abl_nohand", "set_mean", r"\quad leaf encoder without botanical descriptors"),
            ("abl_sam3leaf", "set_mean", r"\quad leaf encoder replaced by the SAM~3 leaf-crop embedding"),
            ("period", "set_mean", r"\quad test specimens paired within capture periods"),
            ("period", "late_sum", r"\quad late fusion, test specimens paired within capture periods")]
    ref = df[(df.variant == "main") & (df.method == "set_mean") & (df.protocol == "grouped") & (df.subset == SUB_ALL)].sort_values(["seed", "fold"]).acc.values
    lines = [r"\begin{table*}[tbp]", r"\centering", r"\footnotesize",
             r"\caption{Multi-organ variety identification: fusion methods and ablation (accuracy, \%, mean $\pm$ SD).}",
             r"\label{tab:fusion_main}", r"\begin{adjustbox}{max width=\textwidth}", r"\begin{tabular}{lccccc}", r"\toprule",
             r" & \multicolumn{2}{c}{Grouped CV} & \multicolumn{2}{c}{Session-level CV} & \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r" & Leaf blade + top & All four organs & Leaf blade + top & All four organs & $p$ \\", r"\midrule",
             r"\multicolumn{6}{l}{\emph{Fusion method}} \\"]
    lines += method_rows
    lines += [r"\midrule", r"\multicolumn{6}{l}{\emph{Ablation of the organ-set model}} \\"]
    for var, mth, lab in rows:
        d = df[(df.variant == var) & (df.method == mth)]
        if not len(d):
            continue
        cells = []
        for proto in ("grouped", "session"):
            for sub in (SUB_LEAF, SUB_ALL):
                v = d[(d.protocol == proto) & (d.subset == sub)].acc
                if var == "period" and proto == "session":
                    cells.append("--")  # session-level test specimens are drawn from one capture period by construction
                    continue
                cells.append(pm(v.mean(), v.std(ddof=1)) if len(v) > 1 else (f"{100*v.mean():.2f}" if len(v) else "--"))
        b = d[(d.protocol == "grouped") & (d.subset == SUB_ALL)].sort_values(["seed", "fold"]).acc.values
        p = corrected_t(ref, b) if (var, mth) != ("main", "set_mean") and len(b) == len(ref) else float("nan")
        if lab is not None:
            lines.append(lab + " & " + " & ".join(cells) + f" & {ptxt(p)}" + r" \\")
        if var == "period":
            M = "Set" if mth == "set_mean" else "Late"
            for sub, S in ((SUB_ALL, "All"), (SUB_LEAF, "Leaf")):
                mac(f"Period{M}{S}G", 100 * d[(d.protocol == "grouped") & (d.subset == sub)].acc.mean())
        if var.startswith("abl_") or (var, mth) == ("main", "set_attn"):
            tag = {"abl_nodrop": "NoDrop", "abl_noaug": "NoAug", "abl_nohand": "NoHand", "abl_sam3leaf": "SamLeaf", "main": "Attn"}[var]
            for proto, P in (("grouped", "G"), ("session", "S")):
                mac(f"Abl{tag}Leaf{P}", 100 * d[(d.protocol == proto) & (d.subset == SUB_LEAF)].acc.mean())
                mac(f"Abl{tag}All{P}", 100 * d[(d.protocol == proto) & (d.subset == SUB_ALL)].acc.mean())
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table*}"]
    (TAB / "tab_fusion_main.tex").write_text("\n".join(lines))


def per_class(df):
    d = df[(df.variant == "main") & (df.method == "set_mean") & (df.subset == SUB_ALL) & (df.protocol == "session")]
    pc = np.array([json.loads(x) if isinstance(x, str) else x for x in d.per_class]).mean(0) * 100
    names = {"CP77-400": "CPA", "CPF-237": "CPFA", "CPF-250": "CPFB", "CPF-251": "CPFC", "CPF-253": "CPFD"}
    for c, v in zip(CLASSES, pc):
        mac("RecSess" + names[c], v, "{:.1f}")  # LaTeX command names cannot contain digits


# --------------------------------------------------------------------------------------- segmentation, efficiency
def segmentation():
    for cat in ORGANS:
        r = pd.read_csv(RESULTS / f"sam3_{cat}.csv")
        mac(f"Sam{SHORT[cat]}Det", 100 * (r.method != "colour_fallback").mean(), "{:.1f}")
        mac(f"Sam{SHORT[cat]}Cov", 100 * r.coverage.median(), "{:.1f}")
    a = pd.read_csv(RESULTS / "mask_agreement_leafwidth.csv")
    s = a[a.method != "colour_fallback"]
    mac("MaskIoUMedian", s.iou.median(), "{:.3f}"); mac("MaskDiceMedian", s.dice.median(), "{:.3f}")
    mac("MaskIoULow", str(int((s.iou < 0.8).sum())), "{}"); mac("MaskIoUN", str(len(s)), "{}")


def efficiency():
    p = RESULTS / "efficiency_organs.json"
    if not p.exists():
        return
    e = json.loads(p.read_text())
    lines = [r"\begin{table}[tbp]", r"\centering", r"\footnotesize",
             r"\caption{Inference cost of the four-organ system per specimen on one GPU.}",
             r"\label{tab:eff_organs}", r"\begin{tabular}{lrr}", r"\toprule", r"Stage & Time (ms) & Parameters (M) \\", r"\midrule"]
    for o in ORGANS:
        lines.append(f"SAM~3 segmentation, {NAME[o].lower()} & {e['sam3_ms_per_image'][o]:.0f} & \\\\")
    lines.append(f"DINOv2 organ encoding (4 crops) & {sum(e['dinov2_ms_per_crop'].values()):.0f} & {e['params_M']['dinov2_s']:.1f} \\\\")
    lines.append(f"Organ-set fusion & {e['fusion_ms_per_specimen']:.1f} & {e['params_M']['organ_set_net']:.2f} \\\\")
    lines.append(r"\midrule")
    lines.append(f"Total per specimen & {e['total_ms_per_specimen']:.0f} & {e['params_M']['sam3'] + e['params_M']['dinov2_s'] + e['params_M']['organ_set_net']:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "tab_eff_organs.tex").write_text("\n".join(lines))
    mac("EffTotalMs", e["total_ms_per_specimen"], "{:.0f}"); mac("EffSamParams", e["params_M"]["sam3"], "{:.0f}")
    mac("EffSetParams", e["params_M"]["organ_set_net"] * 1000, "{:.0f}"); mac("EffPeakMB", e["peak_gpu_memory_MB"], "{:.0f}")
    mac("EffGPU", e["gpu"].replace("NVIDIA ", ""), "{}")


def main():
    integrity()
    for fn in (organ_audit, segmentation, efficiency):
        try:
            fn()
        except FileNotFoundError as err:
            print("skipped", fn.__name__, err)
    try:
        df = load_fusion()
        fusion_ablation(df, fusion_main(df)); fusion_subsets(df); per_class(df)
    except ValueError as err:
        print("no fusion results yet", err)
    (TAB / "numbers_organs.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
    print(len(macros), "macros")


if __name__ == "__main__":
    main()
