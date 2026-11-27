"""Print a one-line summary of every cached run (optionally filtered by a name prefix)."""
import sys, json, glob, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from common import RESULTS

pre = sys.argv[1] if len(sys.argv) > 1 else ""
for f in sorted(glob.glob(str(RESULTS / "runs" / f"{pre}*.json"))):
    d = json.load(open(f)); n = os.path.basename(f)[:-5]; s = d.get("summary")
    if s:
        print("%-48s acc %6.2f±%5.2f  F1 %6.2f  AUC %.4f  AP %.4f" % (n, 100 * s['acc']['mean'], 100 * s['acc']['std'],
              100 * s['macro_f1']['mean'], s['macro_auc']['mean'], s['macro_ap']['mean']))
    else:
        fs = d["folds"]
        line = "  ".join(f"[{f['fold']}] acc {100*f['acc']:.2f} bal {100*f['bal_acc']:.2f} F1 {100*f['macro_f1']:.2f}" for f in fs[:2])
        extra = f"  swap_acc {100*d['swap_acc']:.2f} follow {100*d['follow_rate']:.2f}" if "swap_acc" in d else ""
        print("%-48s %s%s" % (n, line if len(fs) <= 2 else "", extra))
