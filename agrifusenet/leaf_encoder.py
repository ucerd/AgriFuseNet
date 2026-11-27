"""Leaf-blade encoder: channel-wise attention fusion of branch embeddings with modality dropout.

Every branch (masked-blade embedding, blade-tile embedding, botanical descriptors) is projected to k
dimensions; a gating network produces, for every channel, a softmax over the branches, and the fused k-D code is
the attention-weighted sum. During training each branch is removed with probability p_modality (at least one is
kept). mode="scalar" is a two-branch scalar gate, alpha = sigmoid(w^T [u_1; u_2]), for comparison.

LeafEncoder is a scikit-learn estimator: it trains the fusion network on the training data and classifies the
L2-normalised code with an RBF SVM (head="svm") or with the network's softmax head (head="softmax").
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionFusion(nn.Module):
    def __init__(self, dims, k=128, n_classes=5, mode="channel", p_drop=0.3, p_modality=0.2):
        super().__init__()
        self.mode, self.p_modality = mode, p_modality
        if mode == "scalar":
            assert len(dims) == 2, "the scalar gate fuses exactly two branches"
            self.proj = nn.ModuleList([nn.Linear(d, k) for d in dims])
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
            U = torch.stack(u, 1)  # (B, branches, k)
            keep = torch.ones(U.shape[:2], device=U.device)
            if self.training and self.p_modality > 0:
                keep = (torch.rand_like(keep) > self.p_modality).float()
                keep[keep.sum(1) == 0, torch.randint(self.nb, (1,)).item()] = 1  # never drop every branch
            logits = self.gate(torch.cat(u, 1)).view(-1, self.nb, self.k)
            logits = logits.masked_fill(keep[..., None] == 0, -1e4)
            a = logits.softmax(1)  # softmax over branches, separately for every channel
            z = (a * U).sum(1)
            attn = a.mean(2)  # mean branch weight per sample
        return (self.head(z), z, attn) if return_attn else (self.head(z), z)


class LeafEncoder(BaseEstimator, ClassifierMixin):
    """slices: column slices of the input matrix that form the branches."""

    def __init__(self, slices=None, k=128, mode="channel", epochs=150, lr=1e-3, wd=1e-2, batch=128,
                 label_smoothing=0.1, p_drop=0.3, p_modality=0.2, C=10.0, kernel="rbf", head="svm", seed=0):
        self.slices, self.k, self.mode, self.epochs, self.lr, self.wd, self.batch = slices, k, mode, epochs, lr, wd, batch
        self.label_smoothing, self.p_drop, self.p_modality = label_smoothing, p_drop, p_modality
        self.C, self.kernel, self.head, self.seed = C, kernel, head, seed

    def _split(self, X):
        X = torch.as_tensor(self.scaler_.transform(X), dtype=torch.float32, device=DEVICE)
        return [X[:, s] for s in self.slices]

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.classes_ = np.unique(y)
        self.scaler_ = StandardScaler().fit(X)
        xs, yt = self._split(X), torch.as_tensor(y, dtype=torch.long, device=DEVICE)
        net = AttentionFusion([s.stop - s.start for s in self.slices], self.k, len(self.classes_), self.mode,
                              self.p_drop, self.p_modality).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
        steps = self.epochs * int(np.ceil(len(y) / self.batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps)
        net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(y), device=DEVICE)
            for i in range(0, len(y), self.batch):
                b = perm[i:i + self.batch]
                logits, _ = net([x[b] for x in xs])
                loss = F.cross_entropy(logits, yt[b], label_smoothing=self.label_smoothing)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        self.net_ = net.eval()
        if self.head == "svm":
            self.svm_ = SVC(C=self.C, kernel=self.kernel, gamma="scale", decision_function_shape="ovr").fit(self.embed(X), y)
        return self

    @torch.no_grad()
    def embed(self, X, return_attn=False):
        _, z, a = self.net_(self._split(X), return_attn=True)
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


class EmbeddingClassifier:
    """Train a LeafEncoder, then fit any scikit-learn classifier on its k-D code."""

    def __init__(self, encoder, classifier):
        self.encoder, self.classifier = encoder, classifier

    def fit(self, X, y):
        self.encoder.fit(X, y)
        self.classifier.fit(self.encoder.embed(X), y)
        return self

    def predict(self, X):
        return self.classifier.predict(self.encoder.embed(X))

    def predict_proba(self, X):
        return self.classifier.predict_proba(self.encoder.embed(X))
