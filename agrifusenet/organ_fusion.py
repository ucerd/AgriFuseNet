"""Multi-organ variety identification on specimens (one background-free image per organ, same variety).

OrganSetNet encodes every available organ into a k-D token, adds a learned organ-type embedding and pools the
tokens of the organs that are present (masked mean, or one set-attention layer with a class token). It is
permutation-invariant and accepts any non-empty subset of organs. A multi-branch organ (the leaf blade) is
encoded by the channel-attention fusion of leaf_encoder.AttentionFusion.

Training specimens are drawn per variety from training images only; during training each organ is removed with
probability p_organ_drop (organ dropout), so that the model learns to decide from any subset of organs.

Baselines on the same representations and specimens: late fusion (sum of per-organ z-scored SVM margins) and
early fusion (concatenation of the organ features and one SVM).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .leaf_encoder import AttentionFusion

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLASSES = 5


# ----------------------------------------------------------------------------------------------- splits
def organ_splits(data, protocol, seed, n_splits=5):
    """Per-organ outer splits, combined index-wise: list over splits of {organ: (train_idx, test_idx)}.
    No training image of any organ is ever part of a test specimen."""
    per = {}
    for o, d in data.items():
        if protocol == "session":
            per[o] = [(np.nonzero(d["period"] != k)[0], np.nonzero(d["period"] == k)[0]) for k in np.unique(d["period"])]
        else:
            per[o] = list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(d["X"], d["y"], d["g"]))
    K = len(next(iter(per.values())))
    return [{o: per[o][k] for o in data} for k in range(K)]


def draw_specimens(data, idx, n_per_class, rng, pairing="random"):
    """Specimens as rows of image indices, one column per organ (-1 if the organ has no image of the variety).
    pairing='random': organ images drawn independently within the variety;
    pairing='period': all organ images of a specimen come from the same capture period."""
    organs = list(data)
    spec, ys = [], []
    for c in range(N_CLASSES):
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
                    pt = pool[d["period"][pool] == t]
                    rows[sel, j] = rng.choice(pt if len(pt) else pool, sel.sum())
            else:
                rows[:, j] = rng.choice(pool, n_per_class)
        spec.append(rows)
        ys.append(np.full(n_per_class, c))
    return np.concatenate(spec), np.concatenate(ys)


# ----------------------------------------------------------------------------------------------- model
class OrganSetNet(nn.Module):
    def __init__(self, organ_dims, k=128, heads=4, p_drop=0.3, pool="mean"):
        """organ_dims: per organ, the list of branch dimensions (one entry for a single-branch organ)."""
        super().__init__()
        self.pool = pool
        self.enc = nn.ModuleList([AttentionFusion(d, k, N_CLASSES, "channel", p_drop, 0.2) if len(d) > 1 else
                                  nn.Sequential(nn.LayerNorm(d[0]), nn.Dropout(p_drop), nn.Linear(d[0], k), nn.GELU(),
                                                nn.Linear(k, k))
                                  for d in organ_dims])
        self.type_emb = nn.Parameter(torch.randn(len(organ_dims), k) * 0.02)
        if pool == "attn":
            self.cls = nn.Parameter(torch.randn(1, 1, k) * 0.02)
            self.attn = nn.TransformerEncoder(nn.TransformerEncoderLayer(k, heads, 2 * k, p_drop, batch_first=True,
                                                                         norm_first=True), 1)
        self.head = nn.Sequential(nn.LayerNorm(k), nn.Dropout(p_drop), nn.Linear(k, N_CLASSES))

    def encode(self, i, xs):
        e = self.enc[i]
        return e(xs, return_attn=True)[1] if isinstance(e, AttentionFusion) else e(xs[0])

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


def fit_set_model(data, tr, pool="mean", seed=0, use_aug=True, p_organ_drop=0.3, epochs=60, steps=40, batch=256):
    """Train OrganSetNet on specimens drawn from the training images tr = {organ: indices}."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    organs = list(data)
    sc = {o: Scaler(data[o]["X"][tr[o]]) for o in organs}
    views = {}
    for o in organs:
        d = data[o]
        Xs = [d["X"][tr[o]]] + ([d["Xa"][a][tr[o]] for a in range(d["Xa"].shape[0])] if use_aug else [])
        views[o] = torch.as_tensor(sc[o](np.stack(Xs)), dtype=torch.float32, device=DEVICE)  # (views, n_train, d)
    pos = {o: {c: np.nonzero(data[o]["y"][tr[o]] == c)[0] for c in range(N_CLASSES)} for o in organs}
    net = OrganSetNet([[s.stop - s.start for s in data[o]["slices"]] for o in organs], pool=pool).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * steps)
    net.train()
    for _ in range(epochs * steps):
        yb = rng.integers(0, N_CLASSES, batch)
        present = ~(rng.random((batch, len(organs))) < p_organ_drop)
        inputs = []
        for j, o in enumerate(organs):
            has = np.array([len(pos[o][c]) > 0 for c in yb])
            rows = np.array([rng.choice(pos[o][c]) if len(pos[o][c]) else 0 for c in yb])
            present[:, j] &= has
            va = rng.integers(0, views[o].shape[0], batch)
            x = views[o][torch.as_tensor(va, device=DEVICE), torch.as_tensor(rows, device=DEVICE)]
            inputs.append([x[:, s] for s in data[o]["slices"]])
        empty = ~present.any(1)
        present[empty, rng.integers(0, len(organs), empty.sum())] = True
        logits = net(inputs, torch.as_tensor(present, device=DEVICE))
        loss = F.cross_entropy(logits, torch.as_tensor(yb, device=DEVICE), label_smoothing=0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return net.eval(), sc


@torch.no_grad()
def predict_set(net, sc, data, spec, subset):
    """Predicted variety of every specimen using only the organs in subset."""
    organs = list(data)
    inputs, present = [], np.zeros((len(spec), len(organs)), bool)
    for j, o in enumerate(organs):
        rows = np.where(spec[:, j] >= 0, spec[:, j], 0)
        x = torch.as_tensor(sc[o](data[o]["X"][rows]), dtype=torch.float32, device=DEVICE)
        inputs.append([x[:, s] for s in data[o]["slices"]])
        present[:, j] = (spec[:, j] >= 0) & (o in subset)
    present[~present.any(1), 0] = True
    return net(inputs, torch.as_tensor(present, device=DEVICE)).argmax(1).cpu().numpy()


# ----------------------------------------------------------------------------------------------- baselines
def _z(d):
    return (d - d.mean(1, keepdims=True)) / (d.std(1, keepdims=True) + 1e-9)


def organ_scores(data, tr, use_aug=True):
    """Per organ, an RBF SVM trained on that organ's training images; returns z-scored margins of all images."""
    out = {}
    for o, d in data.items():
        Xs = [d["X"][tr[o]]] + ([d["Xa"][a][tr[o]] for a in range(d["Xa"].shape[0])] if use_aug else [])
        m = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")).fit(np.vstack(Xs), np.tile(d["y"][tr[o]], len(Xs)))
        Z = np.zeros((len(d["y"]), N_CLASSES), np.float32)
        Z[:, m.classes_] = _z(m.decision_function(d["X"]))
        out[o] = Z
    return out


def late_fusion(scores, spec, organs, subset):
    tot = sum(np.where(spec[:, [j]] >= 0, scores[o][np.maximum(spec[:, j], 0)], 0)
              for j, o in enumerate(organs) if o in subset)
    return tot.argmax(1)


def early_fusion(data, tr, spec_te, subset, rng, n_train=3000):
    """Concatenated organ features of training specimens -> one RBF SVM."""
    organs = [o for o in data if o in subset]
    spec_tr, y_tr = draw_specimens({o: data[o] for o in organs}, {o: tr[o] for o in organs}, n_train // N_CLASSES, rng)
    sc = {o: Scaler(data[o]["X"][tr[o]]) for o in organs}

    def concat(spec):
        return np.hstack([sc[o](data[o]["X"][np.maximum(spec[:, j], 0)]) * (spec[:, [j]] >= 0)
                          for j, o in enumerate(organs)])
    jj = [list(data).index(o) for o in organs]
    m = SVC(C=10, gamma="scale").fit(concat(spec_tr), y_tr)
    return m.predict(concat(spec_te[:, jj]))
