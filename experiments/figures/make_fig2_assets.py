"""Image assets for Fig. 2 (SAM 3 organ masks): one overlay per organ and variety.

The image of each organ and variety is the SAM 3-segmented image with the median detection score
(figures/fig2_selection.csv), i.e. a representative rather than a best-case example. The organ keeps its
colours, the background is darkened and desaturated, and the mask outline is drawn in yellow.

Writes figures/assets/fig2/<organ>_<variety>.png
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import DATA, CACHE, load_rgb  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets" / "fig2"
OUT.mkdir(parents=True, exist_ok=True)
OUTLINE = (255, 210, 31)


def overlay(a, m, size=600):
    g = a.mean(2, keepdims=True)
    bg = (0.35 * (0.5 * a + 0.5 * g)).astype(np.uint8)  # darkened, half-desaturated background
    o = np.where(m[..., None], a, bg).astype(np.uint8)
    o = cv2.resize(o, (size, size), interpolation=cv2.INTER_AREA)
    mm = cv2.resize(m.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    cnt, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(o, cnt, -1, OUTLINE, 3, lineType=cv2.LINE_AA)
    return o


sel = pd.read_csv(HERE / "fig2_selection.csv")
for r in sel.itertuples():
    a = load_rgb(str(DATA / r.relative_path))
    m = np.asarray(Image.open(CACHE / "masks_sam3" / (r.relative_path.replace("/", "__") + ".png"))) > 0
    Image.fromarray(overlay(a, m)).save(OUT / f"{r.organ}_{r.variety}.png")
print(len(sel), "overlays written to", OUT)
