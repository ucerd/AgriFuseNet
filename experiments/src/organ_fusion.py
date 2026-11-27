"""Multi-organ variety identification: organ-set fusion vs. late (score-sum) and early (concatenation) fusion,
evaluated on pseudo-specimens (one background-free image per organ, same variety).

Organ views (no background pixel reaches any encoder):
  leafwidth   : the AgriFuseNet leaf branches [masked blade | blade tiles | 164-D handcrafted]  (leaf_encoder=agrifuse)
                or the DINOv2 feature of the SAM 3 leaf crop                                    (leaf_encoder=sam3obj)
  leavePattern, buddataset, crossSection : DINOv2 features of the SAM 3 organ crop (organs_obj_dinov2_s.npz)

Outer splits are made per organ (grouped: StratifiedGroupKFold on leakage groups, seed s, fold k; session: capture
thirds) and the k-th split of every organ is combined, so no training image of any organ is ever in a test
specimen. Training specimens are drawn from training images only, test specimens from test images only, with the
same draws for every method (paired comparison).

pairing=random : the organ images of a test specimen are drawn independently within the variety;
pairing=period : all organ images of a test specimen come from the same capture period (third of each organ's
                 capture sequence), a proxy for "photographed together", which correlates session effects.

Usage: python organ_fusion.py <seeds e.g. 0,1> --protocol grouped|session [--variant NAME] [--methods ...]
Writes results/organ_fusion/<variant>_<protocol>_s<seeds>.csv (one row per split x subset x method).
"""
import argparse
import itertools
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from common import DATA, CACHE, RESULTS, CLASSES, load_manifest
from groups import components
from fusion import FusionNet

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NC = len(CLASSES)
ORGANS = ["leafwidth", "leavePattern", "buddataset", "crossSection"]


# ----------------------------------------------------------------------------------------------- data
def load_organs(organs, leaf_encoder="agrifuse", hand=True):
    """Per organ: X (n, d), Xa (n_aug, n, d), y, leakage groups g, capture third, branch slices."""
    full = np.load(CACHE / "organs_dinov2_s.npz", allow_pickle=True)
    frow = {p: i for i, p in enumerate(full["relative_path"])}
    obj = np.load(CACHE / "organs_obj_dinov2_s.npz", allow_pickle=True)
    orow = {p: i for i, p in enumerate(obj["relative_path"])}
    meta = pd.read_csv(DATA / "metadata" / "samples.csv")
    bad = set(meta.relative_path[meta.groupby("hash_sha256").variety.transform("nunique") > 1])
    out = {}
    for o in organs:
        if o == "leafwidth" and leaf_encoder == "agrifuse":
            D = dict(np.load(CACHE / "deep_dinov2_s.npz"))
            H = pd.read_parquet(CACHE / "handcrafted.parquet").values
            sub = load_manifest()
            blocks, ablocks = [D["leaf"], D["tiles"]], [D["aug_leaf"], D["aug_tiles"]]
            if hand:
                blocks.append(H); ablocks.append(np.repeat(H[None], 4, 0))
            X = np.hstack(blocks).astype(np.float32)
            Xa = np.concatenate(ablocks, 2).astype(np.float32)
            edges = np.cumsum([0] + [b.shape[1] for b in blocks])
            slices = [slice(a, b) for a, b in zip(edges[:-1], edges[1:])]
            g = pd.read_csv(CACHE / "groups.csv").group.values  # the paper's leaf groups
        else:
            sub = meta[(meta.feature_type == o) & ~meta.relative_path.isin(bad)].sort_values("relative_path").reset_index(drop=True)
            sub["y"] = sub.variety.map(CLASSES.index)
            ix = np.array([orow[p] for p in sub.relative_path])
            X, Xa = obj["obj"][ix], obj["aug_obj"][:, ix]
            slices = [slice(0, X.shape[1])]
            E = normalize(full["full"][[frow[p] for p in sub.relative_path]])
            g = components(sub, E @ E.T, 0.985, 10)
        y = sub.y.values
        num = sub.relative_path.str.extract(r"img_(\d+)")[0].astype(int).values
        third = np.zeros(len(sub), int)
        for c in np.unique(y):
            jj = np.nonzero(y == c)[0]; jj = jj[np.argsort(num[jj])]
            for k, part in enumerate(np.array_split(jj, 3)):
                third[part] = k
        out[o] = dict(X=X, Xa=Xa, y=y, g=g, third=third, slices=slices)
    return out


def organ_splits(data, protocol, seed):
    """List over k of {organ: (train_idx, test_idx)}."""
    per = {}
    for o, d in data.items():
        if protocol == "session":
            per[o] = [(np.nonzero(d["third"] != k)[0], np.nonzero(d["third"] == k)[0]) for k in range(3)]
        else:
            per[o] = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(d["X"], d["y"], d["g"]))
    K = len(next(iter(per.values())))
    return [{o: per[o][k] for o in data} for k in range(K)]


def draw(data, idx, n_per_class, rng, pairing="random"):
    """Pseudo-specimens: per class, n rows of one image index per organ (-1 if the organ has no such image)."""
    organs = list(data)
    spec, ys = [], []
    for c in range(NC):
        rows = np.full((n_per_class, len(organs)), -1)
        periods = rng.integers(0, 3, n_per_class)
        for j, o in enumerate(organs):
            d = data[o]
            pool = idx[o][d["y"][idx[o]] == c]
            if not len(pool):
                continue
            if pairing == "period":
                for t in range(3):
                    sel = periods == t
                    pt = pool[d["third"][pool] == t]
                    rows[sel, j] = rng.choice(pt if len(pt) else pool, sel.sum())
            else:
                rows[:, j] = rng.choice(pool, n_per_class)
        spec.append(rows); ys.append(np.full(n_per_class, c))
    return np.concatenate(spec), np.concatenate(ys)


# ----------------------------------------------------------------------------------------------- set model
class OrganSetNet(nn.Module):
    """Organ encoders -> organ tokens (+ organ-type embedding) -> pooling over the organs present -> classifier.
    pool='mean': masked mean of the tokens; pool='attn': one set-attention layer with a CLS token and a
    missing-organ key mask. A multi-branch organ (the AgriFuseNet leaf) is encoded by the channel-attention fusion."""

    def __init__(self, organ_dims, k=128, heads=4, p_drop=0.3, pool="mean"):
        super().__init__()
        self.pool = pool
        self.enc = nn.ModuleList([FusionNet(d, k, NC, "channel", p_drop, 0.2) if len(d) > 1 else
                                  nn.Sequential(nn.LayerNorm(d[0]), nn.Dropout(p_drop), nn.Linear(d[0], k), nn.GELU(), nn.Linear(k, k))
                                  for d in organ_dims])
        self.type_emb = nn.Parameter(torch.randn(len(organ_dims), k) * 0.02)
        if pool == "attn":
            self.cls = nn.Parameter(torch.randn(1, 1, k) * 0.02)
            self.attn = nn.TransformerEncoder(nn.TransformerEncoderLayer(k, heads, 2 * k, p_drop, batch_first=True, norm_first=True), 1)
        self.head = nn.Sequential(nn.LayerNorm(k), nn.Dropout(p_drop), nn.Linear(k, NC))

    def encode(self, i, xs):
        e = self.enc[i]
        return e(xs, return_attn=True)[1] if isinstance(e, FusionNet) else e(xs[0])

    def forward(self, organ_inputs, present):
        T = torch.stack([self.encode(i, xs) for i, xs in enumerate(organ_inputs)], 1) + self.type_emb  # (B, O, k)
        if self.pool == "mean":
            w = present.float()[..., None]
            return self.head((T * w).sum(1) / w.sum(1).clamp_min(1))
        B = T.shape[0]
        z = torch.cat([self.cls.expand(B, -1, -1), T], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=T.device), ~present], 1)
        return self.head(self.attn(z, src_key_padding_mask=pad)[:, 0])


class Scaler:
    def __init__(self, X):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6

    def __call__(self, X):
        return (X - self.mu) / self.sd


def fit_set_model(data, tr, pool, seed, use_aug=True, p_organ_drop=0.3, epochs=60, steps=40, batch=256):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    organs = list(data)
    sc = {o: Scaler(data[o]["X"][tr[o]]) for o in organs}
    pool_x = {}
    for o in organs:
        d = data[o]
        Xs = [d["X"][tr[o]]] + ([d["Xa"][a][tr[o]] for a in range(d["Xa"].shape[0])] if use_aug else [])
        pool_x[o] = torch.as_tensor(sc[o](np.stack(Xs)), dtype=torch.float32, device=DEV)  # (views, n_tr, d)
    pos = {o: {c: np.nonzero(data[o]["y"][tr[o]] == c)[0] for c in range(NC)} for o in organs}
    net = OrganSetNet([[s.stop - s.start for s in data[o]["slices"]] for o in organs], pool=pool).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * steps)
    net.train()
    for _ in range(epochs * steps):
        yb = rng.integers(0, NC, batch)
        present = ~(rng.random((batch, len(organs))) < p_organ_drop)
        inputs = []
        for j, o in enumerate(organs):
            has = np.array([len(pos[o][c]) > 0 for c in yb])
            rows = np.array([rng.choice(pos[o][c]) if len(pos[o][c]) else 0 for c in yb])
            present[:, j] &= has
            va = rng.integers(0, pool_x[o].shape[0], batch)
            x = pool_x[o][torch.as_tensor(va, device=DEV), torch.as_tensor(rows, device=DEV)]
            inputs.append([x[:, s] for s in data[o]["slices"]])
        empty = ~present.any(1)
        present[empty, rng.integers(0, len(organs), empty.sum())] = True
        logits = net(inputs, torch.as_tensor(present, device=DEV))
        loss = F.cross_entropy(logits, torch.as_tensor(yb, device=DEV), label_smoothing=0.1)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net.eval(), sc


@torch.no_grad()
def predict_set(net, sc, data, spec, subset):
    organs = list(data)
    inputs, present = [], np.zeros((len(spec), len(organs)), bool)
    for j, o in enumerate(organs):
        rows = np.where(spec[:, j] >= 0, spec[:, j], 0)
        x = torch.as_tensor(sc[o](data[o]["X"][rows]), dtype=torch.float32, device=DEV)
        inputs.append([x[:, s] for s in data[o]["slices"]])
        present[:, j] = (spec[:, j] >= 0) & (o in subset)
    present[~present.any(1), 0] = True
    return net(inputs, torch.as_tensor(present, device=DEV)).argmax(1).cpu().numpy()


# ----------------------------------------------------------------------------------------------- baselines
def zs(d):
    return (d - d.mean(1, keepdims=True)) / (d.std(1, keepdims=True) + 1e-9)


def late_fusion_scores(data, tr, use_aug=True):
    """Per-organ RBF-SVM trained on that organ's training images; organ -> z-scored margins for every image."""
    out = {}
    for o, d in data.items():
        Xs = [d["X"][tr[o]]] + ([d["Xa"][a][tr[o]] for a in range(d["Xa"].shape[0])] if use_aug else [])
        m = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")).fit(np.vstack(Xs), np.tile(d["y"][tr[o]], len(Xs)))
        Z = np.zeros((len(d["y"]), NC), np.float32)
        Z[:, m.classes_] = zs(m.decision_function(d["X"]))
        out[o] = Z
    return out


def early_fusion(data, tr, spec_te, subset, rng, n_train=3000):
    """Concatenate the subset's organ features of a pseudo-specimen and train one RBF-SVM on training specimens."""
    organs = [o for o in data if o in subset]
    spec_tr, y_tr = draw({o: data[o] for o in organs}, {o: tr[o] for o in organs}, n_train // NC, rng)
    sc = {o: Scaler(data[o]["X"][tr[o]]) for o in organs}
    cat = lambda spec: np.hstack([sc[o](data[o]["X"][np.maximum(spec[:, j], 0)]) * (spec[:, [j]] >= 0) for j, o in enumerate(organs)])
    jj = [list(data).index(o) for o in organs]
    m = SVC(C=10, gamma="scale").fit(cat(spec_tr), y_tr)
    return m.predict(cat(spec_te[:, jj]))


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds", nargs="?", default="0")
    ap.add_argument("--protocol", default="grouped")
    ap.add_argument("--variant", default="main")
    ap.add_argument("--methods", default="late_sum,early_concat,set_mean,set_attn")
    ap.add_argument("--leaf-encoder", default="agrifuse")
    ap.add_argument("--no-hand", action="store_true")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--organ-drop", type=float, default=0.3)
    ap.add_argument("--pairing", default="random")
    ap.add_argument("--n-test", type=int, default=200, help="test specimens per class and split")
    a = ap.parse_args()
    methods = a.methods.split(",")
    data = load_organs(ORGANS, a.leaf_encoder, hand=not a.no_hand)
    subsets = {"+".join(c): set(c) for r in range(1, len(ORGANS) + 1) for c in itertools.combinations(ORGANS, r)}
    early_subsets = {"+".join(ORGANS), "leafwidth+leavePattern"}
    rows, t0 = [], time.time()
    seeds = [int(s) for s in a.seeds.split(",")] if a.protocol == "grouped" else [0]
    for s in seeds:
        for k, sp in enumerate(organ_splits(data, a.protocol, s)):
            tr = {o: sp[o][0] for o in ORGANS}; te = {o: sp[o][1] for o in ORGANS}
            rng = np.random.default_rng(1000 + 10 * s + k)
            spec, ys = draw(data, te, a.n_test, rng, a.pairing)
            preds = {}
            if "late_sum" in methods:
                Z = late_fusion_scores(data, tr, not a.no_aug)
                for name, sub in subsets.items():
                    tot = sum(np.where(spec[:, [j]] >= 0, Z[o][np.maximum(spec[:, j], 0)], 0)
                              for j, o in enumerate(ORGANS) if o in sub)
                    preds[("late_sum", name)] = tot.argmax(1)
            if "early_concat" in methods:
                for name in early_subsets:
                    preds[("early_concat", name)] = early_fusion(data, tr, spec, subsets[name], np.random.default_rng(s * 7 + k))
            for pool in ("mean", "attn"):
                if f"set_{pool}" in methods:
                    net, sc = fit_set_model(data, tr, pool, seed=s, use_aug=not a.no_aug, p_organ_drop=a.organ_drop)
                    for name, sub in subsets.items():
                        preds[(f"set_{pool}", name)] = predict_set(net, sc, data, spec, sub)
            for (m, name), p in preds.items():
                rows.append({"variant": a.variant, "protocol": a.protocol, "pairing": a.pairing, "seed": s, "fold": k,
                             "method": m, "subset": name, "n_organs": len(subsets[name]),
                             "acc": float((p == ys).mean()),
                             "per_class": [float((p[ys == c] == c).mean()) for c in range(NC)],
                             "confusion": confusion_matrix(ys, p, labels=list(range(NC))).tolist()})
            allk = "+".join(ORGANS)
            print(f"{a.variant} {a.protocol} s{s} f{k} all-organs: " +
                  " ".join(f"{m}={100*r['acc']:.1f}" for r in rows[-len(preds):] for m in [r['method']] if r["subset"] == allk) +
                  f" ({time.time()-t0:.0f}s)", flush=True)
    out = RESULTS / "organ_fusion"; out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"{a.variant}_{a.protocol}_s{a.seeds.replace(',', '-')}.csv", index=False)
    print("finished-ok", flush=True)


if __name__ == "__main__":
    main()
