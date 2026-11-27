"""Attention fusion modules and an sklearn-style estimator: learned 128-D fused embedding -> SVM.

Branches are given as a list of column slices of one feature matrix, so the estimator plugs into
evalkit.run_cv like any other model.

mode="channel": per-sample, per-channel softmax attention over branches (proposed).
mode="scalar":  the original AgriFuseNet gate, alpha = sigmoid(w^T[u_d; u_h]) over exactly two branches.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FusionNet(nn.Module):
    def __init__(self, dims, k=128, n_classes=5, mode="channel", p_drop=0.3, p_modality=0.2):
        super().__init__()
        self.mode, self.p_modality = mode, p_modality
        if mode == "scalar":
            assert len(dims) == 2
            self.proj = nn.ModuleList([nn.Linear(d, k) for d in dims])  # u = W f + b, as in the manuscript
            self.gate = nn.Linear(2 * k, 1)
        else:
            self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Dropout(p_drop), nn.Linear(d, k), nn.GELU(),
                                                     nn.Linear(k, k)) for d in dims])
            self.gate = nn.Sequential(nn.Linear(len(dims) * k, k), nn.GELU(), nn.Linear(k, len(dims) * k))
        self.k, self.nb = k, len(dims)
        self.head = nn.Sequential(nn.Dropout(p_drop), nn.Linear(k, n_classes))

    def forward(self, xs, return_attn=False):
        u = [p(x) for p, x in zip(self.proj, xs)]
        if self.mode == "scalar":
            a = torch.sigmoid(self.gate(torch.cat(u, 1)))
            z = a * u[0] + (1 - a) * u[1]
            attn = torch.cat([a, 1 - a], 1)
        else:
            U = torch.stack(u, 1)  # (B, nb, k)
            keep = torch.ones(U.shape[:2], device=U.device)
            if self.training and self.p_modality > 0:
                keep = (torch.rand_like(keep) > self.p_modality).float()
                keep[keep.sum(1) == 0, torch.randint(self.nb, (1,)).item()] = 1  # never drop every branch
            logits = self.gate(torch.cat(u, 1)).view(-1, self.nb, self.k)
            logits = logits.masked_fill(keep[..., None] == 0, -1e4)
            a = logits.softmax(1)  # softmax over branches, separately for every channel
            z = (a * U).sum(1)
            attn = a.mean(2)  # (B, nb) average branch weight
        return (self.head(z), z, attn) if return_attn else (self.head(z), z)


class FusionSVM(BaseEstimator, ClassifierMixin):
    """Train FusionNet on the training fold, then fit an SVM on its L2-normalised 128-D embedding."""

    def __init__(self, slices=None, k=128, mode="channel", epochs=150, lr=1e-3, wd=1e-2, batch=128,
                 label_smoothing=0.1, p_drop=0.3, p_modality=0.2, C=10.0, kernel="rbf", head="svm", seed=0):
        self.slices, self.k, self.mode, self.epochs, self.lr, self.wd, self.batch = slices, k, mode, epochs, lr, wd, batch
        self.label_smoothing, self.p_drop, self.p_modality = label_smoothing, p_drop, p_modality
        self.C, self.kernel, self.head, self.seed = C, kernel, head, seed

    def _fit_scaler(self, X):
        self.scaler_ = StandardScaler().fit(X)

    def _split(self, X):
        X = torch.as_tensor(self.scaler_.transform(X), dtype=torch.float32, device=DEVICE)
        return [X[:, s] for s in self.slices]

    def _make_net(self):
        return FusionNet([s.stop - s.start for s in self.slices], self.k, len(self.classes_), self.mode,
                         self.p_drop, self.p_modality)

    def fit(self, X, y):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.classes_ = np.unique(y)
        self._fit_scaler(X)
        xs, yt = self._split(X), torch.as_tensor(y, dtype=torch.long, device=DEVICE)
        net = self._make_net().to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
        steps = self.epochs * int(np.ceil(len(y) / self.batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps)
        net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(y), device=DEVICE)
            for i in range(0, len(y), self.batch):
                b = perm[i:i + self.batch]
                logits, z = net([x[b] for x in xs])
                loss = F.cross_entropy(logits, yt[b], label_smoothing=self.label_smoothing)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        self.net_ = net.eval()
        if self.head == "svm":
            self.svm_ = SVC(C=self.C, kernel=self.kernel, gamma="scale", decision_function_shape="ovr").fit(self.embed(X), y)
        return self

    @torch.no_grad()
    def embed(self, X, return_attn=False):
        logits, z, a = self.net_(self._split(X), return_attn=True)
        z = F.normalize(z, dim=1).cpu().numpy()
        return (z, a.cpu().numpy()) if return_attn else z

    @torch.no_grad()
    def _logits(self, X):
        return self.net_(self._split(X))[0].cpu()

    def predict(self, X):
        if self.head == "svm":
            return self.svm_.predict(self.embed(X))
        return self._logits(X).argmax(1).numpy()

    def decision_function(self, X):
        if self.head == "svm":
            return self.svm_.decision_function(self.embed(X))
        return self._logits(X).numpy()
