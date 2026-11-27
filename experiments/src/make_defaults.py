"""Write tables/numbers_defaults.tex: a visible placeholder for every number macro used in main.tex that the
experiment logs have not defined (yet), so that missing values are never silently omitted."""
import re
from common import ROOT, REV

tex = (ROOT / "main.tex").read_text(encoding="utf8")
used = set(re.findall(r"\\([A-Z][A-Za-z]+)", tex))
known = set()
for f in ("numbers.tex", "numbers_organs.tex"):
    known |= set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (REV / "tables" / f).read_text()))
LATEX = {"Big", "Bigg", "Large", "LARGE", "Huge", "LaTeX", "TeX", "Delta", "Sigma", "Omega", "Gamma", "Lambda"}
missing = sorted(used - known - LATEX)
(REV / "tables" / "numbers_defaults.tex").write_text(
    "% placeholders for values not yet generated\n" + "".join(f"\\providecommand{{\\{m}}}{{\\textbf{{??}}}}\n" for m in missing))
print(len(missing), "placeholders:", " ".join(missing))
