"""Dataset loading and construction of the phased drift scenario (slide 10).

Phases (default):
  P1 Initial environment      Normal + attack group A, original operating condition
  P2 Operating condition drift Same classes, covariate shift applied (concept drift)
  P3 New attack               Normal + attack group B under the shifted condition
  P4 Previous attack returns  Normal + A + B (tests catastrophic forgetting)
Phase 5 (resource-limited edge) is the execution profile of the whole run
(single CPU thread, latency/CPU/memory/energy measured), not a separate stream.
"""
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def make_synthetic(seed=0, n_features=40, n_normal=80000, n_attack=15000):
    """Gaussian-mixture stand-in with IIoT-like class names.

    Similar attacks (DDoS_UDP/DDoS_TCP, SQL_injection/XSS) share centres with a
    small perturbation so the task is not trivially separable. Only meant to
    exercise the pipeline before the real datasets are available.
    """
    rng = np.random.default_rng(10_000 + seed)
    n_comp, n_inf = 3, 10

    def centres_like(base=None):
        if base is None:
            c = rng.normal(0, 0.3, size=(n_comp, n_features))
            inf = rng.choice(n_features, size=n_inf, replace=False)
            c[:, inf] += rng.normal(0, 1.4, size=(n_comp, n_inf))
            return c
        c = base.copy()
        inf = rng.choice(n_features, size=8, replace=False)
        c[:, inf] += rng.normal(0, 1.0, size=(n_comp, 8))
        return c

    c_normal = centres_like()
    c_udp = centres_like()
    c_sql = centres_like()
    spec = {
        "Normal": (c_normal, n_normal),
        "DDoS_UDP": (c_udp, n_attack),
        "DDoS_TCP": (centres_like(c_udp), n_attack),
        "SQL_injection": (c_sql, n_attack),
        "XSS": (centres_like(c_sql), n_attack),
    }
    Xs, ys = [], []
    for name, (centres, n) in spec.items():
        comp = rng.integers(0, n_comp, size=n)
        Xs.append(centres[comp] + rng.normal(0, 1.0, size=(n, n_features)))
        ys.append(np.full(n, name, dtype=object))
    X = np.concatenate(Xs).astype(np.float32)
    y = np.concatenate(ys)
    names = [f"f{i}" for i in range(n_features)]
    return X, y, names, {"source": "synthetic"}


def _to_seconds(s):
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.5:
        return num.to_numpy(float)
    dt = pd.to_datetime(s.astype(str).str.strip(), errors="coerce", format="mixed")
    out = dt.astype("int64").to_numpy(float) / 1e9
    out[dt.isna().to_numpy()] = np.nan
    return out


def load_csv(path, label_col, drop_cols=(), other_label_cols=(), keep_classes=None,
             max_per_class=None, sample_frac=1.0, drop_duplicates=True, seed=0,
             chunksize=250_000, max_categories=50, time_col=None, target_rows=None,
             min_per_class=0):
    """Generic loader for tabular IIoT intrusion datasets (Edge-IIoTset, X-IIoTID, ...).

    - drops identifier / leakage columns (IPs, ports, timestamps, payloads)
    - numeric-like object columns are coerced, low-cardinality strings one-hot encoded,
      high-cardinality strings dropped
    - optional duplicate removal and per-class cap
    - natural-drift mode: `time_col` is parsed to seconds and returned in info["times"]
      (never used as a feature); `target_rows` subsamples uniformly over time, keeping at least
      `min_per_class` rows per class
    """
    rng = np.random.default_rng(seed)
    drop_lower = {c.lower() for c in drop_cols} | {c.lower() for c in other_label_cols}
    keep = set(keep_classes) if keep_classes is not None else None

    # pass 1 (labels only): per-class sampling rate so that large files (e.g. MU-IoT,
    # 35M rows) are subsampled uniformly over the whole file while reading
    rate = None
    if target_rows:
        counts = {}
        for ch in pd.read_csv(path, usecols=lambda c: str(c).strip() == label_col,
                              chunksize=2_000_000):
            for k, v in ch.iloc[:, 0].astype(str).str.strip().value_counts().items():
                counts[k] = counts.get(k, 0) + v
        r = target_rows / sum(counts.values())
        rate = {k: min(1.0, max(r, min_per_class / v)) for k, v in counts.items()}
    elif max_per_class:
        counts = {}
        for ch in pd.read_csv(path, usecols=lambda c: str(c).strip() == label_col,
                              chunksize=2_000_000):
            for k, v in ch.iloc[:, 0].astype(str).str.strip().value_counts().items():
                counts[k] = counts.get(k, 0) + v
        # oversample 1.5x so that duplicate removal still leaves ~max_per_class rows
        rate = {k: min(1.0, 1.5 * max_per_class / v) for k, v in counts.items()}

    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        if label_col not in chunk.columns:
            raise KeyError(f"label column '{label_col}' not in {list(chunk.columns)}")
        chunk[label_col] = chunk[label_col].astype(str).str.strip()
        if keep is not None:
            chunk = chunk[chunk[label_col].isin(keep)]
        if time_col is not None:
            chunk = chunk.assign(__t__=_to_seconds(chunk[time_col]))
        if rate is not None:
            r = chunk[label_col].map(rate).to_numpy(float)
            chunk = chunk[rng.random(len(chunk)) < r]
        if sample_frac < 1.0:
            chunk = chunk.sample(frac=sample_frac, random_state=int(rng.integers(2**31)))
        chunk = chunk[[c for c in chunk.columns
                       if c in (label_col, "__t__") or c.lower() not in drop_lower]]
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    n_raw = len(df)
    if drop_duplicates:
        df = df.drop_duplicates(ignore_index=True)
    if max_per_class:
        idx = []
        for _, rows in df.groupby(label_col).indices.items():
            if len(rows) > max_per_class:
                rows = rng.choice(rows, size=max_per_class, replace=False)
            idx.append(rows)
        df = df.iloc[np.sort(np.concatenate(idx))].reset_index(drop=True)

    times = df.pop("__t__").to_numpy(float) if "__t__" in df else None
    y = df.pop(label_col).to_numpy(dtype=object)
    parts, dropped = [], []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            parts.append(s.astype("float64").rename(col))
            continue
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().mean() >= 0.9:
            parts.append(num.rename(col))
        elif s.nunique(dropna=True) <= max_categories:
            parts.append(pd.get_dummies(s.astype(str), prefix=col, dtype="float64"))
        else:
            dropped.append(col)
    feats = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    const = [c for c in feats.columns if feats[c].nunique() <= 1]
    feats = feats.drop(columns=const)
    info = {"source": str(path), "rows_raw": n_raw, "rows_used": len(feats),
            "n_features": feats.shape[1], "dropped_high_cardinality": dropped,
            "dropped_constant": const,
            "class_counts": pd.Series(y).value_counts().to_dict()}
    if times is not None:
        info["times"] = times
    return feats.to_numpy(np.float32), y, list(feats.columns), info


def load_dataset(cfg, seed=0, keep_classes=None, cache_dir="cache"):
    kind = cfg["type"]
    if kind == "synthetic":
        return make_synthetic(seed=0, n_features=cfg.get("n_features", 40),
                              n_normal=cfg.get("n_normal", 80000),
                              n_attack=cfg.get("n_attack", 15000))
    if kind != "csv":
        raise ValueError(f"unknown dataset type {kind}")
    key = hashlib.sha1(json.dumps([cfg, sorted(keep_classes or []), seed],
                                  sort_keys=True).encode()).hexdigest()[:12]
    cache = Path(cache_dir) / f"{Path(cfg['path']).stem.replace(' ', '_')}_{key}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["X"], z["y"], list(z["names"]), z["info"].item()
    out = load_csv(cfg["path"], cfg["label_col"], cfg.get("drop_cols", ()),
                   cfg.get("other_label_cols", ()), keep_classes=keep_classes,
                   max_per_class=cfg.get("max_per_class"),
                   sample_frac=cfg.get("sample_frac", 1.0),
                   drop_duplicates=cfg.get("drop_duplicates", True), seed=seed,
                   time_col=cfg.get("time_col"), target_rows=cfg.get("target_rows"),
                   min_per_class=cfg.get("min_per_class", 0))
    cache.parent.mkdir(parents=True, exist_ok=True)
    X, y, names, info = out
    np.savez(cache, X=X, y=y, names=np.array(names, dtype=object), info=np.array(info, dtype=object))
    return out


# --------------------------------------------------------------------------
# Scenario
# --------------------------------------------------------------------------
class CovariateShift:
    """Operating-condition change: x' = x * scale + offset on a subset of continuous features.

    Applied in the standardised feature space; emulates workload / firmware /
    configuration changes that move the feature distribution of all classes.
    """

    def __init__(self, d, candidates, rng, frac=0.4, scale_range=(0.5, 2.0), offset_std=1.0):
        k = max(1, int(round(frac * len(candidates))))
        self.features = np.sort(rng.choice(candidates, size=k, replace=False))
        self.scale = np.ones(d, np.float32)
        self.offset = np.zeros(d, np.float32)
        lo, hi = np.log(scale_range[0]), np.log(scale_range[1])
        self.scale[self.features] = np.exp(rng.uniform(lo, hi, size=k))
        self.offset[self.features] = rng.normal(0, offset_std, size=k)

    def __call__(self, X):
        return X * self.scale + self.offset


class Scenario:
    def __init__(self, X, y_str, cfg, seed):
        rng = np.random.default_rng(seed)
        self.cfg = cfg
        self.normal = cfg["normal"]
        self.groups = {"Normal": [self.normal], **cfg["groups"]}
        self.class_names = [c for g in self.groups.values() for c in g]
        present = set(np.unique(y_str))
        missing = [c for c in self.class_names if c not in present]
        if missing:
            raise ValueError(f"classes {missing} not in dataset; available: {sorted(present)}")
        self.cidx = {c: i for i, c in enumerate(self.class_names)}
        self.group_of = {self.cidx[c]: g for g, cs in self.groups.items() for c in cs}
        self.normal_idx = self.cidx[self.normal]
        self.n_classes = len(self.class_names)
        self.phase_names = [p["name"] for p in cfg["phases"]]
        bs = cfg["batch_size"]

        # disjoint pools per class: test / pretrain / stream
        p1_mix = cfg["phases"][0]["mix"]
        pre_counts = self._class_counts(p1_mix, cfg["pretrain_size"])
        self.pools, test_idx, pre_idx = {}, {}, []
        for c in self.class_names:
            idx = rng.permutation(np.flatnonzero(y_str == c))
            n_test = min(cfg["test_per_class"], len(idx) // 5)
            test_idx[c] = idx[:n_test]
            n_pre = min(pre_counts.get(c, 0), (len(idx) - n_test) // 2)
            pre_idx.append(idx[n_test:n_test + n_pre])
            self.pools[c] = idx[n_test + n_pre:]
        # clean, labelled validation sets held by independent validators (update gate)
        # (only when configured, so that streams of runs without a gate are unchanged)
        vcfg = cfg.get("validators", {"n": 0, "per_class": 0})
        self.n_validators = vcfg["n"]
        val_idx = {}
        for c in self.class_names:
            k = min(vcfg["n"] * vcfg["per_class"], len(self.pools[c]) // 10)
            val_idx[c] = np.array_split(self.pools[c][:k], max(1, vcfg["n"]))
            self.pools[c] = self.pools[c][k:]
        pre_idx = np.concatenate(pre_idx)

        # standardise with pretraining statistics only (no leakage)
        self.mu = X[pre_idx].mean(0)
        self.sd = X[pre_idx].std(0)
        self.sd[self.sd < 1e-6] = 1.0
        Z = lambda idx: ((X[idx] - self.mu) / self.sd).astype(np.float32)
        y_int = np.array([self.cidx.get(c, -1) for c in y_str])

        self.X_pre, self.y_pre = Z(pre_idx), y_int[pre_idx]
        self.tests = {g: (Z(np.concatenate([test_idx[c] for c in cs])),
                          y_int[np.concatenate([test_idx[c] for c in cs])])
                      for g, cs in self.groups.items()}

        self.val = [{c: (Z(val_idx[c][v]), y_int[val_idx[c][v]]) for c in self.class_names}
                    for v in range(self.n_validators)]
        # classes that exist in the environment up to (and including) each phase
        seen, self.phase_classes = set(), []
        for ph in cfg["phases"]:
            seen |= {c for g in ph["mix"] for c in self.groups[g]}
            self.phase_classes.append(sorted(self.cidx[c] for c in seen))

        continuous = np.flatnonzero([len(np.unique(X[pre_idx[:5000], j])) > 2
                                     for j in range(X.shape[1])])
        sh = cfg["shift"]
        self.shift = CovariateShift(X.shape[1], continuous, rng, sh["frac_features"],
                                    sh["scale_range"], sh["offset_std"])

        # build the stream
        cursor = {c: 0 for c in self.class_names}
        self.stream, self.phase_start, warned = [], [], set()
        for p_i, ph in enumerate(cfg["phases"]):
            self.phase_start.append(len(self.stream))
            for _ in range(ph["batches"]):
                counts = self._sample_counts(ph["mix"], bs, rng)
                idx = []
                for c, n in counts.items():
                    pool = self.pools[c]
                    if cursor[c] + n > len(pool):
                        if c not in warned:
                            warnings.warn(f"stream pool of '{c}' exhausted; reshuffling (samples reused)")
                            warned.add(c)
                        self.pools[c] = pool = rng.permutation(pool)
                        cursor[c] = 0
                    idx.append(pool[cursor[c]:cursor[c] + n])
                    cursor[c] += n
                idx = rng.permutation(np.concatenate(idx))
                Xb = Z(idx)
                if ph.get("shift", False):
                    Xb = self.shift(Xb)
                self.stream.append((Xb.astype(np.float32), y_int[idx], p_i))
        self.phase_end = self.phase_start[1:] + [len(self.stream)]
        self.drift_points = self.phase_start[1:]

    def _class_counts(self, mix, n):
        out = {}
        for g, w in mix.items():
            for c in self.groups[g]:
                out[c] = int(round(n * w / sum(mix.values()) / len(self.groups[g])))
        return out

    def _sample_counts(self, mix, n, rng):
        gs = list(mix)
        w = np.array([mix[g] for g in gs], float)
        out = {}
        for g, k in zip(gs, rng.multinomial(n, w / w.sum())):
            members = self.groups[g]
            for c, kc in zip(members, rng.multinomial(k, np.ones(len(members)) / len(members))):
                if kc:
                    out[c] = out.get(c, 0) + kc
        return out

    def test_set(self, group, phase_idx):
        X, y = self.tests[group]
        if phase_idx >= 0 and self.cfg["phases"][phase_idx].get("shift", False):
            X = self.shift(X)
        return X, y

    def validator_sets(self, phase_idx):
        """Per-validator clean (X, y) for classes present so far, under the phase's operating condition."""
        names = [self.class_names[i] for i in self.phase_classes[phase_idx]]
        shift = self.cfg["phases"][phase_idx].get("shift", False)
        out = []
        for v in self.val:
            X = np.concatenate([v[c][0] for c in names])
            y = np.concatenate([v[c][1] for c in names])
            out.append((self.shift(X) if shift else X, y))
        return out

    def describe(self):
        lines = [f"classes: {self.class_names}",
                 f"features: {self.X_pre.shape[1]}, pretrain: {len(self.y_pre)}, "
                 f"stream batches: {len(self.stream)} x {self.cfg['batch_size']}",
                 f"phases: {list(zip(self.phase_names, self.phase_start))}",
                 f"shifted features: {len(self.shift.features)}"]
        return "\n".join(lines)
