"""Edge IDS backbones. All take a flat feature vector (B, d) and return class logits.

Records in the IIoT datasets are per-flow/per-packet feature vectors, not time series.
Sequence models therefore read the vector as a short sequence of feature patches
(`patch` consecutive features per step); the Transformer tokenises each feature
individually (FT-Transformer style).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import device as dev


class MLP(nn.Module):
    def __init__(self, d, n_classes, hidden=(128, 64), **_):
        super().__init__()
        layers, prev = [], d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    def __init__(self, d, n_classes, channels=(32, 64), pooled=8, **_):
        super().__init__()
        c1, c2 = channels
        self.conv = nn.Sequential(
            nn.Conv1d(1, c1, 3, padding=1), nn.ReLU(),
            nn.Conv1d(c1, c2, 3, padding=1), nn.ReLU(),
            nn.AdaptiveMaxPool1d(pooled))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c2 * pooled, 64), nn.ReLU(),
                                  nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.conv(x.unsqueeze(1)))


def _patches(x, patch):
    """(B, d) -> (B, ceil(d/patch), patch), zero-padded."""
    b, d = x.shape
    pad = (-d) % patch
    if pad:
        x = F.pad(x, (0, pad))
    return x.view(b, -1, patch)


class RNN(nn.Module):
    def __init__(self, d, n_classes, cell="lstm", hidden=64, patch=8, **_):
        super().__init__()
        self.patch = patch
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(patch, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        out, _ = self.rnn(_patches(x, self.patch))
        return self.head(out[:, -1])


class CNNLSTM(nn.Module):
    """Conv feature extractor over the feature axis, LSTM over the pooled conv sequence."""

    def __init__(self, d, n_classes, channels=32, hidden=64, pooled=12, **_):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, channels, 3, padding=1), nn.ReLU(),
                                  nn.AdaptiveMaxPool1d(pooled))
        self.rnn = nn.LSTM(channels, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        h = self.conv(x.unsqueeze(1)).transpose(1, 2)  # (B, pooled, channels)
        out, _ = self.rnn(h)
        return self.head(out[:, -1])


class FTTransformer(nn.Module):
    """Feature tokenizer + Transformer encoder with a [CLS] token (Gorishniy et al., 2021)."""

    def __init__(self, d, n_classes, d_token=32, layers=2, heads=4, ff=64, dropout=0.0, **_):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d, d_token))
        self.bias = nn.Parameter(torch.empty(d, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        layer = nn.TransformerEncoderLayer(d_token, heads, ff, dropout, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(), nn.Linear(d_token, n_classes))

    def forward(self, x):
        tok = x.unsqueeze(-1) * self.weight + self.bias          # (B, d, d_token)
        tok = torch.cat([self.cls.expand(len(x), -1, -1), tok], 1)
        return self.head(self.encoder(tok)[:, 0])

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))
        nn.init.zeros_(self.cls)


ARCHS = {
    "mlp": lambda d, k, c: MLP(d, k, **c),
    "cnn": lambda d, k, c: CNN1D(d, k, **c),
    "lstm": lambda d, k, c: RNN(d, k, cell="lstm", **c),
    "gru": lambda d, k, c: RNN(d, k, cell="gru", **c),
    "cnn_lstm": lambda d, k, c: CNNLSTM(d, k, **c),
    "transformer": lambda d, k, c: FTTransformer(d, k, **c),
}


def build_model(arch, d, n_classes, arch_cfg=None):
    return ARCHS[arch](d, n_classes, dict(arch_cfg or {}))


def reset_parameters(model):
    for m in model.modules():
        if m is model:
            continue
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
        elif hasattr(m, "_reset_parameters"):  # nn.MultiheadAttention
            m._reset_parameters()
    if hasattr(model, "reset_parameters"):
        model.reset_parameters()


def param_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def train_epochs(model, opt, X, y, epochs, batch, rng):
    """Plain supervised training; returns number of gradient steps."""
    Xt = torch.from_numpy(X).to(dev.DEVICE)
    yt = torch.from_numpy(y.astype(np.int64)).to(dev.DEVICE)
    steps = 0
    model.train()
    for _ in range(epochs):
        for s in np.array_split(rng.permutation(len(X)), max(1, len(X) // batch)):
            s = torch.from_numpy(s).to(dev.DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(Xt[s]), yt[s]).backward()
            opt.step()
            steps += 1
    return steps
