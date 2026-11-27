"""Image assets for Fig. 1 (pipeline overview): one specimen of CPF-237.

Writes figures/assets/: raw_<organ>.png (photograph), crop_<organ>.png (SAM 3 organ crop, non-organ pixels grey),
blade_masked.png (masked blade), blade_band.png (rectified blade band) and tile_<k>.png (four blade tiles).
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import DATA, CACHE, load_rgb  # noqa: E402
from segment import load_mask  # noqa: E402
from views import leaf_view, rectify, leaf_tiles  # noqa: E402
from organs_obj_extract import obj_view  # noqa: E402

OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(exist_ok=True)
SPEC = {"leafwidth": "leafwidth/CPF-237/img_021.jpg", "leavePattern": "leavePattern/CPF-237/img_021.jpg",
        "buddataset": "buddataset/CPF-237/img_021.jpg", "crossSection": "crossSection/CPF-237/img_021.jpg"}


def save(a, name, size=None):
    im = Image.fromarray(np.ascontiguousarray(a))
    if size:
        im = im.resize(size, Image.LANCZOS)
    im.save(OUT / name)


for organ, rel in SPEC.items():
    a = load_rgb(str(DATA / rel))
    m = np.asarray(Image.open(CACHE / "masks_sam3" / (rel.replace("/", "__") + ".png"))) > 0
    save(a, f"raw_{organ}.png", (320, 320))
    save(obj_view(a, m), f"crop_{organ}.png", (320, 320))

a = load_rgb(str(DATA / SPEC["leafwidth"]))
m = load_mask(SPEC["leafwidth"])
save(leaf_view(a, m), "blade_masked.png", (320, 320))
band, _ = rectify(a, m)
save(band, "blade_band.png")
for k, t in enumerate(leaf_tiles(a, m)):
    save(t, f"tile_{k}.png", (160, 160))
print("assets written to", OUT)
