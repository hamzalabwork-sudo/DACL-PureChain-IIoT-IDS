"""Natural-drift scenario: replay real traffic in its recorded time order (no injected drift).

Protocol (fixed for every dataset, not tuned):
  - rows without a valid timestamp and classes with < `min_class_rows` rows are dropped
  - a random `test_frac` of each class (at most `test_per_class` rows), drawn from the whole
    timeline, is held out for checkpoint evaluation
  - the remaining rows are sorted by time; the first `pretrain_frac` of the timeline is the
    deployment-time training set, the rest is streamed in order in batches of `batch_size`
  - attack classes seen before deployment form group A (known); attack classes that first
    appear in the stream form group B (new). Each first appearance of a B class is a natural
    drift event (used for detection delay, exposure window and MAE)
  - the stream is split into `n_segments` equal segments for checkpoint evaluation
Exposes the same interface as data.Scenario, so learners, metrics and reports are unchanged.
"""
import numpy as np


class NaturalScenario:
    def __init__(self, X, y_str, times, cfg, seed):
        rng = np.random.default_rng(seed)
        bs = cfg["batch_size"]
        ok = ~np.isnan(times)
        X, y_str, times = X[ok], y_str[ok], times[ok]
        names, counts = np.unique(y_str, return_counts=True)
        keep_names = set(names[counts >= cfg.get("min_class_rows", 200)])
        keep = np.isin(y_str, list(keep_names))
        X, y_str, times = X[keep], y_str[keep], times[keep]

        # held-out checkpoint test sets, drawn from the whole timeline
        test_mask = np.zeros(len(y_str), bool)
        for c in keep_names:
            idx = np.flatnonzero(y_str == c)
            k = min(cfg["test_per_class"], int(len(idx) * cfg.get("test_frac", 0.15)))
            test_mask[rng.choice(idx, k, replace=False)] = True

        order = np.flatnonzero(~test_mask)
        order = order[np.argsort(times[order], kind="stable")]
        n_pre = int(len(order) * cfg["pretrain_frac"])
        pre, rest = order[:n_pre], order[n_pre:]

        self.normal = cfg["normal"]
        attacks_pre = sorted(set(y_str[pre]) - {self.normal})
        # B classes ordered by first appearance in the stream
        first = {}
        for i, c in enumerate(y_str[rest]):
            if c != self.normal and c not in attacks_pre and c not in first:
                first[c] = i
        attacks_new = sorted(first, key=first.get)
        self.groups = {"Normal": [self.normal], "A": attacks_pre, "B": attacks_new}
        self.class_names = [self.normal] + attacks_pre + attacks_new
        self.cidx = {c: i for i, c in enumerate(self.class_names)}
        self.group_of = {self.cidx[c]: g for g, cs in self.groups.items() for c in cs}
        self.normal_idx = 0
        self.n_classes = len(self.class_names)
        y_int = np.array([self.cidx[c] for c in y_str])

        self.mu = X[pre].mean(0)
        self.sd = X[pre].std(0)
        self.sd[self.sd < 1e-6] = 1.0
        Z = lambda idx: ((X[idx] - self.mu) / self.sd).astype(np.float32)
        self.X_pre, self.y_pre = Z(pre), y_int[pre]

        n_batches = len(rest) // bs
        rest = rest[:n_batches * bs]
        self.stream = []
        seg = np.array_split(np.arange(n_batches), cfg["n_segments"])
        seg_of = np.empty(n_batches, int)
        for s_i, b in enumerate(seg):
            seg_of[b] = s_i
        for b in range(n_batches):
            idx = rest[b * bs:(b + 1) * bs]
            self.stream.append((Z(idx), y_int[idx], int(seg_of[b])))
        self.phase_start = [int(b[0]) for b in seg]
        self.phase_end = self.phase_start[1:] + [n_batches]
        self.phase_names = [f"S{i + 1}" for i in range(cfg["n_segments"])]
        self.cfg = {**cfg, "phases": [{"name": n, "shift": False} for n in self.phase_names]}

        # natural drift events: batch of first appearance of each new attack class
        dps = sorted({first[c] // bs for c in attacks_new if first[c] // bs < n_batches})
        self.drift_points = dps
        self.drift_windows = list(zip(dps, dps[1:] + [n_batches]))
        self.first_seen = {c: first[c] // bs for c in attacks_new}

        self.tests = {g: (Z(np.flatnonzero(test_mask & np.isin(y_str, cs))),
                          y_int[np.flatnonzero(test_mask & np.isin(y_str, cs))])
                      for g, cs in self.groups.items() if cs}
        self.groups = {g: cs for g, cs in self.groups.items() if cs}
        seen, self.phase_classes = set(np.unique(self.y_pre).tolist()), []
        for s_i in range(cfg["n_segments"]):
            for X_b, y_b, p in self.stream:
                if p == s_i:
                    seen |= set(np.unique(y_b).tolist())
            self.phase_classes.append(sorted(seen))
        self.n_validators = 0
        self.time_span = (float(times[order].min()), float(times[order].max()))

    def test_set(self, group, phase_idx):
        return self.tests[group]

    def validator_sets(self, phase_idx):
        raise NotImplementedError("the update gate is evaluated on the injected scenario")

    def describe(self):
        return "\n".join([
            f"natural drift: classes {self.class_names}",
            f"known (A): {self.groups.get('A')}, new (B, by first appearance): {self.groups.get('B')}",
            f"features: {self.X_pre.shape[1]}, pretrain: {len(self.y_pre)}, "
            f"stream batches: {len(self.stream)} x {self.cfg['batch_size']}",
            f"drift events (first appearance batch): {self.first_seen}"])
