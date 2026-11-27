"""Generate the LaTeX result tables of the revised manuscript directly from results/runs/*.json.

python make_tables.py [best_backbone]   -> revision/tables/*.tex  (+ tables/numbers.json for in-text values)
Missing runs are skipped with a note, so the script can be rerun as experiments finish.
"""
import sys
import json
import numpy as np
import pandas as pd
from common import RESULTS, REV, CACHE, CLASSES, load_manifest
from evalkit import corrected_ttest

RUNS = RESULTS / "runs"
TAB = REV / "tables"
TAB.mkdir(exist_ok=True)
BEST = sys.argv[1] if len(sys.argv) > 1 else "dinov2_s"
BB_NAME = {"resnet50": "ResNet50", "dinov2_s": "DINOv2 ViT-S/14", "effnet_b0": "EfficientNet-B0",
           "convnext_t": "ConvNeXt-T", "mobilenetv3": "MobileNetV3-L"}
NUM = {}


def run(name):
    p = RUNS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def pm(r, key="acc", scale=100, d=2):
    if r is None:
        return "--"
    s = r["summary"][key]
    return f"{scale*s['mean']:.{d}f} $\\pm$ {scale*s['std']:.{d}f}"


def m(r, key, scale=100, d=2):
    return "--" if r is None else f"{scale*r['summary'][key]['mean']:.{d}f}"


def fold_vals(r, key="acc"):
    return np.array([f[key] for f in sorted(r["folds"], key=lambda f: (f["seed"], f["fold"]))])


def pval(a, b, key="acc"):
    if a is None or b is None:
        return "--"
    n = 981
    _, p = corrected_ttest(fold_vals(a, key), fold_vals(b, key), n_train=0.8 * n, n_test=0.2 * n)
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def write(name, body, caption, label, cols, size="\\small", star=False):
    env = "table*" if star else "table"
    tex = (f"\\begin{{{env}}}[tbp]\n\\centering\n{size}\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
           f"\\begin{{adjustbox}}{{max width=\\textwidth}}\n"
           f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{body}\\bottomrule\n\\end{{tabular}}\n\\end{{adjustbox}}\n\\end{{{env}}}\n")
    (TAB / f"{name}.tex").write_text(tex, encoding="utf-8")
    print("wrote", name)


# ------------------------------------------------------------------------------------------- dataset
def t_dataset():
    df = load_manifest("leafwidth")
    G = pd.read_csv(CACHE / "groups.csv")
    df["group"], df["board"] = G.group.values, G.board.values
    rows = ""
    tot = dict(n=0, u=0, d=0, g=0, s=0, q=0)
    for c in CLASSES:
        x = df[df.variety == c]
        u = x.hash_sha256.nunique()
        vals = dict(n=len(x), u=u, d=len(x) - u, g=x.group.nunique(), s=int((x.board == 0).sum()), q=int((x.board == 1).sum()))
        for k in tot:
            tot[k] += vals[k]
        rows += f"{c} & {vals['n']} & {vals['u']} & {vals['d']} & {vals['g']} & {vals['s']} & {vals['q']} \\\\\n"
    rows += "\\midrule\n" + f"Total & {tot['n']} & {tot['u']} & {tot['d']} & {tot['g']} & {tot['s']} & {tot['q']} \\\\\n"
    NUM["dataset"] = tot
    head = ("Variety & Images & Unique & Exact dupl. & Leaf groups & Striped board & Square board \\\\\n\\midrule\n")
    write("tab_dataset", head + rows,
          "Composition of the leaf-blade category of Sugarcane-PK v1.0.",
          "tab:dataset", "lcccccc")


# ------------------------------------------------------------------------------------------- leakage
def t_leakage():
    rows = ""
    items = [("ResNet50, full image", "leak_{}_resnet50_full"),
             ("34-D global handcrafted, full image", "leak_{}_legacy34"),
             ("Leaf-only 164-D handcrafted (ours)", "leak_{}_hand164"),
             (f"{BB_NAME[BEST]}, full image", f"proto_{{}}_{BEST}_deep_full"),
             (f"Leaf-blade encoder ({BB_NAME[BEST]})", f"proto_{{}}_{BEST}_proposed")]
    for label, pat in items:
        r_rand = run(pat.format("random"))
        r_dup = run(pat.format("groupdup"))
        r_grp = run(pat.format("group"))
        drop = "--" if r_rand is None or r_grp is None else \
            f"{100*(r_rand['summary']['acc']['mean'] - r_grp['summary']['acc']['mean']):.2f}"
        rows += f"{label} & {pm(r_rand)} & {pm(r_dup)} & {pm(r_grp)} & {drop} \\\\\n"
    head = ("Representation (SVM) & Random 5-fold & Duplicate-grouped & Sequence-grouped & $\\Delta$ (pp) \\\\\n\\midrule\n")
    write("tab_leakage", head + rows,
          "Effect of the splitting protocol on accuracy (\\%, mean $\\pm$ SD).",
          "tab:leakage", "lcccc", star=True)


# ------------------------------------------------------------------------------------------- audit
def t_audit():
    rows = ""
    for b in ("resnet50", "dinov2_s", "convnext_t", "effnet_b0", "mobilenetv3"):
        rs = [run(f"audit_{b}_{v}") for v in ("full", "bg", "bgpatch", "leaf", "tiles")]
        if all(r is None for r in rs):
            continue
        rows += BB_NAME[b] + " & " + " & ".join(pm(r) for r in rs) + " \\\\\n"
        if b == BEST:
            NUM["audit_best"] = {v: (r["summary"]["acc"]["mean"] if r else None)
                                 for v, r in zip(("full", "bg", "bgpatch", "leaf", "tiles"), rs)}
    r1, r2 = run("audit_legacy34_fullimage"), run("audit_hand164_leafonly")
    rows += "\\midrule\n" + f"Handcrafted & {pm(r1)} & -- & -- & {pm(r2)} & -- \\\\\n"
    head = ("Features & Full image & Background (leaf masked) & Background patches only & Leaf only & Rectified leaf tiles \\\\\n"
            "\\midrule\n")
    write("tab_audit", head + rows,
          "Accuracy (\\%) of leaf-blade image views for five backbones.",
          "tab:audit", "lccccc", star=True)


# ------------------------------------------------------------------------------------------- comparison
def t_compare_both():
    """Compact main-text comparison: accuracy and p-value for both backbones side by side."""
    items = [("Deep, full image", "cmp_{}_deep_full", "full"),
             ("Deep, full image + PCA-128", "cmp_{}_deep_full_pca128", "full"),
             ("Deep, full image + LDA-4", "cmp_{}_deep_full_lda4", "full"),
             ("34-D global handcrafted", "cmp_legacy34", "full"),
             ("Concat. deep + 34-D handcrafted", "cmp_{}_concat_full_legacy", "full"),
             ("Concat. deep + 34-D handcrafted + PCA-128", "cmp_{}_concat_full_legacy_pca128", "full"),
             ("Scalar-gated attention fusion", "cmp_{}_orig_scalar_fusion", "full"),
             ("Deep, leaf only", "cmp_{}_deep_leaf", "leaf"),
             ("Concat. leaf + tiles + 164-D handcrafted", "cmp_{}_concat_leaf_tiles_hand", "leaf"),
             ("Concat. leaf + tiles + 164-D handcrafted + PCA-128", "cmp_{}_concat_leaf_tiles_hand_pca128", "leaf"),
             ("\\textbf{Leaf-blade encoder}", "cmp_{}_proposed", "leaf")]
    props = {b: run(f"cmp_{b}_proposed") for b in ("resnet50", BEST)}
    rows, last = "", None
    for label, pat, inp in items:
        if last == "full" and inp == "leaf":
            rows += "\\midrule\n"
        last = inp
        cells = []
        for b in ("resnet50", BEST):
            name = pat.format(b)
            r = run(name)
            cells += [pm(r), "--" if r is None or name == f"cmp_{b}_proposed" else pval(props[b], r)]
        rows += f"{label} & {'Full' if inp == 'full' else 'Leaf'} & " + " & ".join(cells) + " \\\\\n"
    head = (" & & \\multicolumn{2}{c}{ResNet50} & \\multicolumn{2}{c}{" + BB_NAME[BEST] + "} \\\\\n"
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
            "Representation & Input & Accuracy (\\%) & $p$ & Accuracy (\\%) & $p$ \\\\\n\\midrule\n")
    write("tab_compare", head + rows, "Comparison of leaf-blade representations (accuracy, mean $\\pm$ SD).",
          "tab:compare", "llcccc", size="\\footnotesize", star=True)


def t_compare(b):
    prop = run(f"cmp_{b}_proposed")
    items = [("Deep, full image", f"cmp_{b}_deep_full", "full"),
             ("Deep, full image + PCA-128", f"cmp_{b}_deep_full_pca128", "full"),
             ("Deep, full image + LDA-4", f"cmp_{b}_deep_full_lda4", "full"),
             ("34-D global handcrafted (GLCM, LBP, colour)", "cmp_legacy34", "full"),
             ("Concat. deep + 34-D handcrafted", f"cmp_{b}_concat_full_legacy", "full"),
             ("Concat. deep + 34-D handcrafted + PCA-128", f"cmp_{b}_concat_full_legacy_pca128", "full"),
             ("Scalar-gated attention fusion (128-D)", f"cmp_{b}_orig_scalar_fusion", "full"),
             ("Deep, leaf only", f"cmp_{b}_deep_leaf", "leaf"),
             ("Concat. leaf + tiles + 164-D handcrafted", f"cmp_{b}_concat_leaf_tiles_hand", "leaf"),
             ("Concat. leaf + tiles + 164-D handcrafted + PCA-128", f"cmp_{b}_concat_leaf_tiles_hand_pca128", "leaf"),
             ("\\textbf{Leaf-blade encoder (128-D)}", f"cmp_{b}_proposed", "leaf")]
    rows, last = "", None
    for label, name, inp in items:
        r = run(name)
        if r is None:
            continue
        if last == "full" and inp == "leaf":
            rows += "\\midrule\n"
        last = inp
        dim = "128" if "fusion" in label or "Proposed" in label or "128" in label else ("4" if "LDA" in label else str(r.get("dim") or "--"))
        p = "--" if name == f"cmp_{b}_proposed" else pval(prop, r)
        rows += (f"{label} & {'Full' if inp == 'full' else 'Leaf'} & {dim} & {pm(r)} & {pm(r, 'macro_f1')} & "
                 f"{m(r, 'macro_auc', 1, 4)} & {m(r, 'macro_ap', 1, 4)} & {p} \\\\\n")
    head = ("Representation & Input & Dim. & Accuracy (\\%) & Macro-F1 (\\%) & Macro-AUC & Macro-AP & $p$ vs.\\ leaf-blade encoder \\\\\n"
            "\\midrule\n")
    write(f"tab_compare_{b}", head + rows,
          f"Comparison of leaf-blade representations with the {BB_NAME[b]} backbone (mean $\\pm$ SD over 25 splits).",
          f"tab:compare_{b}", "llcccccc", size="\\footnotesize", star=True)
    if prop:
        NUM[f"proposed_{b}"] = {k: prop["summary"][k]["mean"] for k in ("acc", "macro_f1", "macro_auc", "macro_ap", "bal_acc", "kappa")}
        NUM[f"proposed_{b}"]["acc_std"] = prop["summary"]["acc"]["std"]
        NUM[f"proposed_{b}"]["acc_ci95"] = prop["summary"]["acc"]["ci95"]


# ------------------------------------------------------------------------------------------- robustness
def t_robust():
    rows = ""
    labels = {"deep_full": "Deep, full image", "legacy34": "34-D global handcrafted (full image)",
              "deep_leaf": "Deep, leaf only", "hand164": "164-D leaf handcrafted", "proposed": "\\textbf{Leaf-blade encoder}"}
    for b in ("resnet50", "dinov2_s"):
        block = ""
        for k, lab in labels.items():
            if k in ("legacy34", "hand164") and b != "resnet50":
                continue  # backbone-independent rows are listed once
            xb, sw = run(f"xboard_{b}_{k}"), run(f"swap_{b}_{k}")
            if xb is None and sw is None:
                continue
            if xb:
                f = sorted(xb["folds"], key=lambda f: f["fold"])
                x0, x1 = f"{100*f[0]['bal_acc']:.2f}", f"{100*f[1]['bal_acc']:.2f}"
                NUM.setdefault("xboard", {})[f"{b}_{k}"] = [f[0]["bal_acc"], f[1]["bal_acc"]]
            else:
                x0 = x1 = "--"
            if sw and "swap_acc" in sw:
                s0, s1 = f"{100*sw['swap_acc']:.2f}", f"{100*sw['follow_rate']:.2f}"
                NUM.setdefault("swap", {})[f"{b}_{k}"] = [sw["swap_acc"], sw["follow_rate"]]
            else:
                s0 = s1 = "--"
            shared = "--" if k in ("legacy34", "hand164") else BB_NAME[b]
            block += f"{lab} & {shared} & {x0} & {x1} & {s0} & {s1} \\\\\n"
        if block:
            rows += block + "\\midrule\n"
    rows = rows[: -len("\\midrule\n")] if rows.endswith("\\midrule\n") else rows
    head = ("Representation & Backbone & Striped$\\rightarrow$square & Square$\\rightarrow$striped & Swap accuracy & "
            "Follows pasted board \\\\\n\\midrule\n")
    write("tab_robust", head + rows,
          "Robustness of leaf-blade representations to the acquisition background (\\%).",
          "tab:robust", "llcccc", size="\\footnotesize", star=True)


# ------------------------------------------------------------------------------------------- ablation
def t_ablation(b):
    ref = f"abl_{b}_reference" if run(f"abl_{b}_reference") else f"cmp_{b}_proposed"
    base = run(ref)
    items = [("Full model (leaf + tiles + handcrafted, channel attention, ModDrop, aug., SVM)", ref),
             ("\\quad without train-time augmentation", f"abl_{b}_proposed_noaug"),
             ("\\quad without modality dropout", f"abl_{b}_proposed_nomodalitydrop"),
             ("\\quad softmax head instead of SVM", f"abl_{b}_proposed_softmaxhead"),
             ("\\quad $k=32$", f"abl_{b}_proposed_k32"), ("\\quad $k=64$", f"abl_{b}_proposed_k64"),
             ("\\quad $k=256$", f"abl_{b}_proposed_k256"),
             ("\\quad leaf + handcrafted (channel attention)", f"abl_{b}_proposed_leaf+hand"),
             ("\\quad leaf + handcrafted (scalar gate)", f"abl_{b}_scalar_leaf+hand"),
             ("\\quad tiles + handcrafted (channel attention)", f"abl_{b}_proposed_tiles+hand"),
             ("\\quad tiles + handcrafted (scalar gate)", f"abl_{b}_scalar_tiles+hand"),
             ("\\quad leaf + tiles, no handcrafted (channel attention)", f"abl_{b}_proposed_leaf+tiles"),
             ("\\quad leaf + tiles, no handcrafted (scalar gate)", f"abl_{b}_scalar_leaf+tiles")]
    items += [(f"\\quad handcrafted without {g}", f"abl_{b}_proposed_minus_{g}")
              for g in ("col", "glcm", "lbp", "gab", "ven", "mor")]
    names = {"col": "colour", "glcm": "GLCM", "lbp": "LBP", "gab": "Gabor", "ven": "venation spectrum", "mor": "morphometrics"}
    rows = ""
    for label, name in items:
        r = run(name)
        if r is None:
            continue
        for g, nm in names.items():
            label = label.replace(f"without {g}", f"without {nm}")
        p = "--" if name == ref else pval(base, r)
        rows += f"{label} & {pm(r)} & {pm(r, 'macro_f1')} & {p} \\\\\n"
    head = "Variant & Accuracy (\\%) & Macro-F1 (\\%) & $p$ vs.\\ full \\\\\n\\midrule\n"
    write(f"tab_ablation_{b}", head + rows,
          f"Ablation of the leaf-blade encoder ({BB_NAME[b]}).",
          f"tab:ablation", "lccc", size="\\footnotesize", star=True)


def t_classifiers(b):
    ref = f"abl_{b}_reference" if run(f"abl_{b}_reference") else f"cmp_{b}_proposed"
    items = [("SVM (RBF)", ref), ("Softmax head", f"abl_{b}_proposed_softmaxhead"),
             ("$k$-NN ($k=5$)", f"clf_{b}_knn"), ("Random forest (300 trees)", f"clf_{b}_rf"),
             ("Decision tree", f"clf_{b}_dt"), ("Gaussian naive Bayes", f"clf_{b}_nb")]
    rows = "".join(f"{lab} & {pm(r)} & {pm(r, 'macro_f1')} & {m(r, 'predict_ms_per_img', 1, 3)} \\\\\n"
                   for lab, r in ((l, run(n)) for l, n in items) if r is not None)
    head = "Classifier on fused 128-D embedding & Accuracy (\\%) & Macro-F1 (\\%) & Predict (ms/img) \\\\\n\\midrule\n"
    write(f"tab_classifiers_{b}", head + rows,
          f"Classifiers trained on the 128-D leaf-blade code ({BB_NAME[b]}).", "tab:classifiers", "lccc", size="\\small")


def t_backbones():
    eff = json.loads((RESULTS / "efficiency.json").read_text()) if (RESULTS / "efficiency.json").exists() else {}
    rows = ""
    for b in ("resnet50", "dinov2_s", "convnext_t", "effnet_b0", "mobilenetv3"):
        full = run(f"cmp_{b}_deep_full") or run(f"bb_{b}_deep_full")
        prop = run(f"cmp_{b}_proposed") or run(f"bb_{b}_proposed")
        if full is None and prop is None:
            continue
        e = eff.get(b, {})
        lat = e.get("median_ms", {}).get("proposed", {}).get("total")
        rows += (f"{BB_NAME[b]} & {e.get('backbone_params_M', '--')} & {pm(full)} & {pm(prop)} & {pm(prop, 'macro_f1')} & "
                 f"{'--' if lat is None else f'{lat:.0f}'} \\\\\n")
    head = ("Backbone & Params (M) & Deep full image & Leaf-blade encoder acc. & Macro-F1 & Latency (ms) \\\\\n"
            "\\midrule\n")
    write("tab_backbones", head + rows,
          "Effect of the frozen backbone on the leaf-blade encoder.", "tab:backbones", "lccccc",
          size="\\footnotesize", star=True)


def t_efficiency():
    p = RESULTS / "efficiency.json"
    if not p.exists():
        return
    eff = json.loads(p.read_text())
    rows = ""
    for b, e in eff.items():
        mb, mp = e["median_ms"]["baseline"], e["median_ms"]["proposed"]
        rows += (f"{BB_NAME[b]} & {mb['total']:.0f} & {mp['segmentation']:.0f} & {mp['views']:.0f} & {mp['backbone']:.0f} & "
                 f"{mp['handcrafted']:.0f} & {mp['fusion+svm']:.1f} & {mp['total']:.0f} & "
                 f"{e['backbone_MB'] + e['fusion_MB'] + e['svm_proposed_MB']:.1f} & "
                 f"{'--' if e.get('process_peak_MB') is None else '%.0f' % e['process_peak_MB']} \\\\\n")
    head = ("Backbone & Baseline total & Segm. & Views & Backbone ($\\times$5) & Handcrafted & Fusion+SVM & Encoder total & "
            "Model size (MB) & Peak RAM (MB) \\\\\n\\midrule\n")
    write("tab_efficiency", head + rows,
          "Inference cost of the leaf-blade encoder per image on a laptop CPU (ms).",
          "tab:efficiency", "lccccccccc", size="\\footnotesize", star=True)
    NUM["efficiency"] = eff


def t_perclass(b):
    r = run(f"cmp_{b}_proposed")
    if r is None:
        return
    s = r["summary"]
    cm = np.array(s["confusion_mean"])
    spec = [(cm.sum() - cm[i].sum() - cm[:, i].sum() + cm[i, i]) / (cm.sum() - cm[i].sum()) for i in range(5)]
    rows = "".join(f"{c} & {100*s['per_class_precision_mean'][i]:.2f} & {100*s['per_class_recall_mean'][i]:.2f} & "
                   f"{100*s['per_class_f1_mean'][i]:.2f} & {100*spec[i]:.2f} \\\\\n" for i, c in enumerate(CLASSES))
    head = "Variety & Precision (\\%) & Recall (\\%) & F1 (\\%) & Specificity (\\%) \\\\\n\\midrule\n"
    write(f"tab_perclass_{b}", head + rows,
          f"Per-variety performance of the leaf-blade encoder ({BB_NAME[b]}).", "tab:perclass", "lcccc", size="\\small")
    NUM[f"confusion_{b}"] = cm.round(1).tolist()


def macros():
    """In-text numbers as LaTeX macros (tables/numbers.tex), so prose and tables cannot disagree."""
    M = {}

    def acc(name, key="acc", sd=False):
        r = run(name)
        if r is None:
            return None
        s = r["summary"][key]
        return f"{100*s['std']:.2f}" if sd else f"{100*s['mean']:.2f}"

    d = NUM.get("dataset", {})
    M.update(NImages=d.get("n"), NUnique=d.get("u"), NDup=d.get("d"), NGroups=d.get("g"))
    for tag, b in (("Best", BEST), ("Res", "resnet50")):
        for v in ("full", "bg", "bgpatch", "leaf", "tiles"):
            M[f"Audit{tag}{v.capitalize()}"] = acc(f"audit_{b}_{v}")
        M[f"Full{tag}"] = acc(f"cmp_{b}_deep_full")
        M[f"FullPCA{tag}"] = acc(f"cmp_{b}_deep_full_pca128")
        M[f"Concat{tag}"] = acc(f"cmp_{b}_concat_full_legacy")
        M[f"Scalar{tag}"] = acc(f"cmp_{b}_orig_scalar_fusion")
        M[f"Leaf{tag}"] = acc(f"cmp_{b}_deep_leaf")
        M[f"ConcatLeaf{tag}"] = acc(f"cmp_{b}_concat_leaf_tiles_hand")
        M[f"Prop{tag}"] = acc(f"cmp_{b}_proposed")
        M[f"Prop{tag}SD"] = acc(f"cmp_{b}_proposed", sd=True)
        M[f"Prop{tag}Fone"] = acc(f"cmp_{b}_proposed", "macro_f1")
        r = run(f"cmp_{b}_proposed")
        s = r["summary"] if r else None
        M[f"Prop{tag}AUC"] = s and f"{s['macro_auc']['mean']:.3f}"
        M[f"Prop{tag}AP"] = s and f"{s['macro_ap']['mean']:.3f}"
        M[f"Prop{tag}Kappa"] = s and f"{s['kappa']['mean']:.3f}"
        M[f"Prop{tag}CIlo"], M[f"Prop{tag}CIhi"] = (f"{100*x:.2f}" for x in s["acc"]["ci95"]) if s else (None, None)
        for k, nm in (("deep_full", "Full"), ("proposed", "Prop"), ("legacy34", "Legacy"), ("deep_leaf", "Leaf"),
                      ("hand164", "Hand")):
            xb = NUM.get("xboard", {}).get(f"{b}_{k}")
            M[f"Xb{nm}{tag}A"], M[f"Xb{nm}{tag}B"] = (f"{100*xb[0]:.2f}", f"{100*xb[1]:.2f}") if xb else (None, None)
            sw = NUM.get("swap", {}).get(f"{b}_{k}")
            M[f"Swap{nm}{tag}Acc"], M[f"Swap{nm}{tag}Follow"] = (f"{100*sw[0]:.2f}", f"{100*sw[1]:.2f}") if sw else (None, None)
    M["LegacyFull"] = acc("audit_legacy34_fullimage")
    M["HandLeaf"] = acc("audit_hand164_leafonly")
    for proto in ("random", "group"):
        M[f"Leak{proto.capitalize()}Res"] = acc(f"leak_{proto}_resnet50_full")
        M[f"Leak{proto.capitalize()}Legacy"] = acc(f"leak_{proto}_legacy34")
    e = NUM.get("efficiency", {}).get(BEST)
    M["LatBase"] = e and f"{e['median_ms']['baseline']['total']:.0f}"
    M["LatProp"] = e and f"{e['median_ms']['proposed']['total']:.0f}"
    M["LatSeg"] = e and f"{e['median_ms']['proposed']['segmentation']:.0f}"
    M["LatHand"] = e and f"{e['median_ms']['proposed']['handcrafted']:.0f}"
    M["LatBackbone"] = e and f"{e['median_ms']['proposed']['backbone']:.0f}"
    M["SizeProp"] = e and f"{e['backbone_MB'] + e['fusion_MB'] + e['svm_proposed_MB']:.1f}"
    M["FusionParamsK"] = e and f"{e['fusion_params_K']:.0f}"
    M["PeakMem"] = e and "process_peak_MB" in e and f"{e['process_peak_MB']:.0f}" or None
    em = NUM.get("efficiency", {}).get("mobilenetv3")
    M["LatPropMobile"] = em and f"{em['median_ms']['proposed']['total']:.0f}"
    M["PropMobile"] = acc("bb_mobilenetv3_proposed")
    M["PropEffnet"] = acc("bb_effnet_b0_proposed")
    M["PropConvnext"] = acc("bb_convnext_t_proposed")
    M["DedupProp"] = acc(f"dedup_{BEST}_proposed")
    M["DedupFull"] = acc(f"dedup_{BEST}_deep_full")
    lines = ["% generated by revision/src/make_tables.py -- do not edit by hand"]
    missing = r"\textbf{??}"
    for k, v in M.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{missing if v is None else v}}}")
    (TAB / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote numbers.tex with", len(M), "macros")


if __name__ == "__main__":
    t_dataset(); t_leakage(); t_audit()
    for bb in ("resnet50", BEST):
        t_compare(bb)
    t_compare_both()
    t_robust(); t_ablation(BEST); t_classifiers(BEST); t_backbones(); t_efficiency(); t_perclass(BEST)
    (TAB / "numbers.json").write_text(json.dumps(NUM, indent=1, default=float))
    macros()
