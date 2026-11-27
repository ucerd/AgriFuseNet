"""Shared paths and dataset loading for the AgriFuseNet-Sugarcane revision experiments."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Dataset_Zenodo_18619387" / "Sugarcane_Varieties_Dataset"
REV = ROOT / "revision"
CACHE = REV / "cache"
RESULTS = REV / "results"
CLASSES = ["CP77-400", "CPF-237", "CPF-250", "CPF-251", "CPF-253"]

for p in (CACHE, RESULTS):
    p.mkdir(parents=True, exist_ok=True)


def load_manifest(feature="leafwidth"):
    """Return the manifest rows of one morphological category, sorted for determinism."""
    df = pd.read_csv(DATA / "metadata" / "samples.csv")
    df = df[df.feature_type == feature].copy()
    df["path"] = df.relative_path.map(lambda p: str(DATA / p))
    df["y"] = df.variety.map(CLASSES.index)
    return df.sort_values("relative_path").reset_index(drop=True)


def load_rgb(path, size=None):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if size is not None:
        im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im)
