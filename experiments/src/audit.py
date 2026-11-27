"""Dataset audit: counts, image properties, pre-augmented images, exact and near duplicates."""
import json
import numpy as np
from PIL import Image
from common import load_manifest, RESULTS, CLASSES

df = load_manifest("leafwidth")
out = {"n_images": len(df), "per_class": df.variety.value_counts().reindex(CLASSES).to_dict(),
       "manifest_split": df.groupby(["variety", "split"]).size().unstack(fill_value=0).to_dict()}

sizes, modes, exif, black_corner, dh = [], [], [], [], []
for p in df.path:
    im = Image.open(p)
    sizes.append(im.size); modes.append(im.mode); exif.append(len(im.getexif()) > 0)
    a = np.asarray(im.convert("RGB")).astype(int)
    k = 24  # corner patch size; rotated augmentations leave pure-black triangles in the corners
    corners = [a[:k, :k], a[:k, -k:], a[-k:, :k], a[-k:, -k:]]
    black_corner.append(sum((c.max(axis=2) < 12).mean() > 0.6 for c in corners))
    g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR)).astype(int)
    dh.append(np.packbits((g[:, 1:] > g[:, :-1]).ravel()))
dh = np.unpackbits(np.stack(dh), axis=1)

out["sizes"] = {str(k): int(v) for k, v in zip(*np.unique([str(s) for s in sizes], return_counts=True))}
out["modes"] = sorted(set(modes))
out["n_with_exif"] = int(np.sum(exif))
out["n_black_corner_images(>=2 corners)"] = int(np.sum(np.array(black_corner) >= 2))
out["black_corner_files"] = df.relative_path[np.array(black_corner) >= 2].tolist()

# exact duplicates by SHA-256
dup = df[df.duplicated("hash_sha256", keep=False)].sort_values("hash_sha256")
out["exact_duplicate_files"] = dup.relative_path.tolist()

# near duplicates by 64-bit difference hash (Hamming distance)
ham = (dh[:, None, :] != dh[None, :, :]).sum(-1)
iu = np.triu_indices(len(df), 1)
for t in (0, 4, 8):
    m = ham[iu] <= t
    same = (df.y.values[iu[0]] == df.y.values[iu[1]])[m]
    out[f"dhash_pairs_le{t}"] = {"total": int(m.sum()), "same_class": int(same.sum()), "cross_class": int((~same).sum())}
close = [(df.relative_path[i], df.relative_path[j], int(ham[i, j])) for i, j in zip(*iu) if ham[i, j] <= 4]
out["dhash_close_pairs_examples"] = close[:40]

(RESULTS / "audit_leafwidth.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if "examples" not in k and "files" not in k}, indent=1))
print("black-corner files:", out["black_corner_files"][:20])
print("exact dups:", out["exact_duplicate_files"][:20])
print("close pairs:", close[:20])
