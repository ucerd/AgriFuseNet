"""Build the flat journal-submission folder from main.tex and supplementary.tex.

Editorial systems do not support sub-directories, so every \\input is inlined, the number macros are replaced
by their values, figures are copied next to the manuscript, and the bibliography is reduced to cited entries.

Usage (from the project root):  python revision/tools/make_submission.py
"""
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "submission"
FIGURES = {"fig_pipeline": "Fig1", "fig_sam3_masks": "Fig2", "fig_leak": "Fig3",
           "fig_audit": "Fig4", "fig_robust": "Fig5", "fig_organs": "Fig6"}
NUMBER_FILES = ["revision/tables/numbers.tex", "revision/tables/numbers_organs.tex"]


def load_macros():
    macros = {}
    for f in NUMBER_FILES:
        for line in (ROOT / f).read_text(encoding="utf-8").splitlines():
            m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", line)
            if m:
                macros[m.group(1)] = m.group(2)
    return macros


def strip_comment_lines(text):
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("%"))


def flatten(src, macros):
    text = (ROOT / src).read_text(encoding="utf-8")
    for f in NUMBER_FILES:
        text = text.replace("\\input{%s}\n" % f, "")

    def inline(m):
        return strip_comment_lines((ROOT / m.group(1)).read_text(encoding="utf-8")).rstrip("\n")
    text = re.sub(r"\\input\{(revision/tables/[^}]+\.tex)\}", inline, text)

    def expand(m):
        name = m.group(1)
        return macros[name] if name in macros else m.group(0)
    text = re.sub(r"\\([A-Za-z]+)\{\}", lambda m: macros.get(m.group(1), m.group(0)), text)
    text = re.sub(r"\\([A-Za-z]+)(?![A-Za-z])", expand, text)
    leftover = [n for n in macros if "\\" + n in text]
    assert not leftover, leftover

    for stem, new in FIGURES.items():
        text = text.replace("revision/figures/%s.png" % stem, "%s.png" % new)
    assert "revision/" not in text, "unresolved path"
    return strip_comment_lines(text) + "\n"


def bib_entries(path):
    text = path.read_text(encoding="utf-8")
    entries, i = {}, 0
    while True:
        i = text.find("@", i)
        if i < 0:
            return entries
        j = text.index("{", i)
        key = text[j + 1:text.index(",", j)].strip()
        depth, k = 0, j
        while True:
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        entries[key] = text[i:k + 1]
        i = k + 1


def run(cmd):
    subprocess.run(cmd, cwd=OUT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    macros = load_macros()
    (OUT / "main.tex").write_text(flatten("main.tex", macros), encoding="utf-8")
    (OUT / "supplementary.tex").write_text(flatten("supplementary.tex", macros), encoding="utf-8")
    for stem, new in FIGURES.items():
        shutil.copy(ROOT / "revision/figures" / f"{stem}.png", OUT / f"{new}.png")
    shutil.copy(ROOT / "elsarticle.cls", OUT)
    bst = subprocess.run(["kpsewhich", "elsarticle-num-names.bst"], capture_output=True, text=True).stdout.strip()
    shutil.copy(bst, OUT)
    shutil.copy(ROOT / "highlights.txt", OUT)

    # first pass with the full bibliography, then keep only the cited entries
    shutil.copy(ROOT / "references.bib", OUT / "references.bib")
    run(["pdflatex", "-interaction=nonstopmode", "main.tex"])
    cited = set()
    for m in re.finditer(r"\\citation\{([^}]+)\}", (OUT / "main.aux").read_text(encoding="utf-8")):
        cited.update(k.strip() for k in m.group(1).split(","))
    entries = bib_entries(ROOT / "references.bib")
    missing = cited - set(entries)
    assert not missing, missing
    (OUT / "references.bib").write_text("\n\n".join(entries[k] for k in entries if k in cited) + "\n", encoding="utf-8")
    run(["bibtex", "main"])
    for _ in range(3):
        run(["pdflatex", "-interaction=nonstopmode", "main.tex"])
    run(["pdflatex", "-interaction=nonstopmode", "supplementary.tex"])
    for f in OUT.iterdir():
        if f.suffix in {".aux", ".log", ".out", ".blg", ".spl", ".fls", ".fdb_latexmk", ".synctex"}:
            f.unlink()
    print("submission folder:", OUT)
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name:28s} {f.stat().st_size / 1024:9.1f} kB")


if __name__ == "__main__":
    main()
