"""Hybrid drift detection on batch-level statistics.

Three signals, each monitored by a one-sided CUSUM (Page-Hinkley form) on a
self-standardised statistic:
  error    supervised: batch error rate (labels revealed after prediction)
  entropy  unsupervised: mean predictive entropy (model uncertainty)
  feature  unsupervised: mean standardised shift of feature means vs. reference
An alarm on any signal triggers adaptation. Severity in (0, 1] is the mean
clipped z-score of the firing signal over its last two batches, divided by the clip.
"""
import numpy as np


class CUSUM:
    def __init__(self, delta=0.5, threshold=10.0, warmup=8, min_std=0.01, clip=6.0):
        self.delta, self.threshold, self.warmup = delta, threshold, warmup
        self.min_std, self.clip = min_std, clip
        self.reset()

    def reset(self):
        self.buf, self.mu, self.sd, self.g, self.z = [], None, None, 0.0, []

    def update(self, x):
        if self.mu is None:
            self.buf.append(x)
            if len(self.buf) >= self.warmup:
                self.mu = float(np.mean(self.buf))
                self.sd = max(float(np.std(self.buf, ddof=1)), self.min_std)
            return False
        z = float(np.clip((x - self.mu) / self.sd, -self.clip, self.clip))
        self.z.append(z)
        self.g = max(0.0, self.g + z - self.delta)
        return self.g > self.threshold

    def severity(self):
        return float(np.clip(np.mean(self.z[-2:]) / self.clip, 0.0, 1.0)) if self.z else 0.0


class FeatureShift:
    """Reference mean/std from the first `warmup` batches, then a scalar distance per batch."""

    def __init__(self, warmup=8):
        self.warmup = warmup
        self.reset()

    def reset(self):
        self.buf, self.mu, self.sd = [], None, None

    def distance(self, X):
        if self.mu is None:
            self.buf.append(X)
            if len(self.buf) >= self.warmup:
                R = np.concatenate(self.buf)
                self.mu, self.sd = R.mean(0), R.std(0) + 1e-3
                self.buf = []
            return None
        return float(np.mean(np.abs(X.mean(0) - self.mu) / self.sd))


class HybridDriftDetector:
    def __init__(self, signals=("error", "entropy", "feature"), delta=0.5, threshold=10.0,
                 warmup=8, clip=6.0, **_):
        self.signals = list(signals)
        self.cusum = {s: CUSUM(delta, threshold, warmup, 0.01, clip) for s in self.signals}
        self.feat = FeatureShift(warmup)
        self.last_fired = []

    def reset(self):
        for c in self.cusum.values():
            c.reset()
        self.feat.reset()

    def update(self, X, err, ent):
        values = {"error": err, "entropy": ent}
        if "feature" in self.signals:
            values["feature"] = self.feat.distance(X)
        fired = [s for s in self.signals
                 if values[s] is not None and self.cusum[s].update(values[s])]
        self.last_fired = fired
        if fired:
            return True, max(self.cusum[s].severity() for s in fired)
        return False, 0.0
