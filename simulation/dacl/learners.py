"""Stream learners: baselines and the proposed DACL.

Protocol per batch (prequential, test-then-train):
  probs = learner.predict_proba(X)      # real-time inference
  learner.observe(X, y, probs)          # labels revealed afterwards
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F

from . import device as dev
from .drift import HybridDriftDetector
from .models import param_bytes, reset_parameters, train_epochs


class ClassBalancedMemory:
    """Fixed-budget replay memory: equal quota per seen class, reservoir sampling within a class."""

    def __init__(self, capacity, d, rng):
        self.capacity, self.d, self.rng = capacity, d, rng
        self.store, self.seen = {}, {}

    def add(self, X, y):
        for c in np.unique(y):
            c = int(c)
            Xc = X[y == c]
            if c not in self.store:
                self.store[c] = np.empty((0, self.d), np.float32)
                self.seen[c] = 0
                q = self.capacity // len(self.store)
                for k, buf in self.store.items():
                    if len(buf) > q:
                        self.store[k] = buf[self.rng.choice(len(buf), q, replace=False)]
            q = self.capacity // len(self.store)
            buf, n = self.store[c], self.seen[c]
            free = q - len(buf)
            if free > 0:
                buf = np.concatenate([buf, Xc[:free]])
                n += len(Xc[:free])
                Xc = Xc[free:]
            if len(Xc):
                j = (self.rng.random(len(Xc)) * (n + np.arange(1, len(Xc) + 1))).astype(np.int64)
                m = j < len(buf)
                buf[j[m]] = Xc[m]
                n += len(Xc)
            self.store[c], self.seen[c] = buf, n

    def sample(self, n):
        classes = [c for c, b in self.store.items() if len(b)]
        cs = self.rng.choice(classes, size=n)
        X = np.empty((n, self.d), np.float32)
        y = np.empty(n, np.int64)
        for c in classes:
            m = cs == c
            if m.any():
                X[m] = self.store[c][self.rng.integers(0, len(self.store[c]), m.sum())]
                y[m] = c
        return X, y

    def classes(self):
        return {c for c, b in self.store.items() if len(b)}

    def nbytes(self):
        return sum(b.nbytes for b in self.store.values())


def _t(X, y=None):
    Xt = torch.from_numpy(X).to(dev.DEVICE)
    return Xt if y is None else (Xt, torch.from_numpy(y.astype(np.int64)).to(dev.DEVICE))


class Learner:
    uses_detector = False

    def __init__(self, model, cfg, rng, scenario):
        self.model, self.cfg, self.rng = model, cfg, rng
        self.opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        self.alarms, self.alarm_info = [], []
        self.t, self.grad_steps, self.extra_bytes = 0, 0, 0

    @torch.no_grad()
    def predict_proba(self, X):
        self.model.eval()
        return F.softmax(self.model(_t(X)), 1).cpu().numpy()

    def observe(self, X, y, probs):
        self.t += 1

    def _step(self, loss):
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.grad_steps += 1

    def state_bytes(self):
        return param_bytes(self.model) + self.extra_bytes


class Static(Learner):
    """Train once, deploy, never update."""


class FineTune(Learner):
    """Naive online fine-tuning on every labelled batch, no retention mechanism."""

    def observe(self, X, y, probs):
        self.model.train()
        Xt, yt = _t(X, y)
        for _ in range(self.cfg["finetune_steps"]):
            self._step(F.cross_entropy(self.model(Xt), yt))
        self.t += 1


class ExperienceReplay(Learner):
    """Continuous online update mixing every batch with a class-balanced replay sample."""

    def __init__(self, model, cfg, rng, scenario):
        super().__init__(model, cfg, rng, scenario)
        self.mem = ClassBalancedMemory(cfg["memory_size"], scenario.X_pre.shape[1], rng)
        self.mem.add(scenario.X_pre, scenario.y_pre)

    def observe(self, X, y, probs):
        self.model.train()
        for _ in range(self.cfg["er_steps"]):
            Xr, yr = self.mem.sample(len(X))
            Xt, yt = _t(np.concatenate([X, Xr]), np.concatenate([y, yr]))
            self._step(F.cross_entropy(self.model(Xt), yt))
        self.mem.add(X, y)
        self.extra_bytes = self.mem.nbytes()
        self.t += 1


class OnlineEWC(Learner):
    """Online EWC: per-batch fine-tuning with a Fisher-weighted anchor, consolidated periodically."""

    def __init__(self, model, cfg, rng, scenario):
        super().__init__(model, cfg, rng, scenario)
        e = cfg["ewc"]
        self.lam, self.gamma, self.every = e["lambda"], e["gamma"], e["consolidate_every"]
        self.fisher, self.anchor, self.window = None, None, []
        n = min(2048, len(scenario.y_pre))
        sel = rng.choice(len(scenario.y_pre), n, replace=False)
        self._consolidate(scenario.X_pre[sel], scenario.y_pre[sel])

    def _consolidate(self, X, y):
        # train mode: cuDNN RNN backward requires it (no dropout/batch-norm in our models,
        # so this is numerically identical to eval mode)
        self.model.train()
        params = [p for p in self.model.parameters()]
        fisher = [torch.zeros_like(p) for p in params]
        Xt, yt = _t(X, y)
        chunks = np.array_split(np.arange(len(X)), max(1, len(X) // 32))
        for s in chunks:
            self.model.zero_grad()
            F.cross_entropy(self.model(Xt[s]), yt[s]).backward()
            for f, p in zip(fisher, params):
                f += p.grad.detach() ** 2 / len(chunks)
        self.fisher = fisher if self.fisher is None else [
            self.gamma * fo + fn for fo, fn in zip(self.fisher, fisher)]
        self.anchor = [p.detach().clone() for p in params]
        self.extra_bytes = 2 * param_bytes(self.model)

    def observe(self, X, y, probs):
        self.model.train()
        Xt, yt = _t(X, y)
        penalty = sum((f * (p - a) ** 2).sum()
                      for f, p, a in zip(self.fisher, self.model.parameters(), self.anchor))
        self._step(F.cross_entropy(self.model(Xt), yt) + 0.5 * self.lam * penalty)
        self.window.append((X, y))
        self.t += 1
        if self.t % self.every == 0:
            Xw = np.concatenate([w[0] for w in self.window[-5:]])
            yw = np.concatenate([w[1] for w in self.window[-5:]])
            self._consolidate(Xw, yw)
            self.window = []


class LwF(Learner):
    """Learning without Forgetting (online): per-batch fine-tuning + distillation from a snapshot
    of the model refreshed every `snapshot_every` batches. No replay memory."""

    def __init__(self, model, cfg, rng, scenario):
        super().__init__(model, cfg, rng, scenario)
        c = cfg["lwf"]
        self.every, self.lam, self.T = c["snapshot_every"], c["lambda_kd"], c["temperature"]
        self.teacher = copy.deepcopy(model).eval()
        self.extra_bytes = param_bytes(model)

    def observe(self, X, y, probs):
        self.model.train()
        Xt, yt = _t(X, y)
        out = self.model(Xt)
        with torch.no_grad():
            t = self.teacher(Xt)
        kd = F.kl_div(F.log_softmax(out / self.T, 1), F.softmax(t / self.T, 1),
                      reduction="batchmean") * self.T ** 2
        self._step(F.cross_entropy(out, yt) + self.lam * kd)
        self.t += 1
        if self.t % self.every == 0:
            self.teacher = copy.deepcopy(self.model).eval()


class DERpp(Learner):
    """Dark Experience Replay++ (Buzzega et al., NeurIPS 2020): reservoir buffer of
    (x, y, logits); loss = CE(new) + alpha * MSE(logits on replay, stored logits) + beta * CE(replay)."""

    def __init__(self, model, cfg, rng, scenario):
        super().__init__(model, cfg, rng, scenario)
        c = cfg["derpp"]
        self.alpha, self.beta, self.cap = c["alpha"], c["beta"], cfg["memory_size"]
        d = scenario.X_pre.shape[1]
        self.bX = np.empty((0, d), np.float32)
        self.by = np.empty(0, np.int64)
        self.bz = np.empty((0, scenario.n_classes), np.float32)
        self.n_seen = 0
        sel = rng.choice(len(scenario.y_pre), min(self.cap, len(scenario.y_pre)), replace=False)
        self._add(scenario.X_pre[sel], scenario.y_pre[sel], self._logits(scenario.X_pre[sel]))

    @torch.no_grad()
    def _logits(self, X):
        self.model.eval()
        return self.model(_t(X)).cpu().numpy()

    def _add(self, X, y, z):
        free = self.cap - len(self.by)
        if free > 0:  # fill
            self.bX = np.concatenate([self.bX, X[:free]])
            self.by = np.concatenate([self.by, y[:free]])
            self.bz = np.concatenate([self.bz, z[:free]])
            self.n_seen += len(X[:free])
            X, y, z = X[free:], y[free:], z[free:]
        if len(X):  # reservoir sampling
            j = (self.rng.random(len(X)) * (self.n_seen + np.arange(1, len(X) + 1))).astype(np.int64)
            m = j < self.cap
            self.bX[j[m]], self.by[j[m]], self.bz[j[m]] = X[m], y[m], z[m]
            self.n_seen += len(X)
        self.extra_bytes = self.bX.nbytes + self.bz.nbytes

    def _replay(self, idx):
        Xr, yr = _t(self.bX[idx], self.by[idx])
        return self.model(Xr), yr

    def observe(self, X, y, probs):
        self.model.train()
        Xt, yt = _t(X, y)
        out = self.model(Xt)
        i1 = self.rng.integers(0, len(self.by), len(X))
        i2 = self.rng.integers(0, len(self.by), len(X))
        loss = (F.cross_entropy(out, yt)
                + self.alpha * F.mse_loss(self.model(_t(self.bX[i1])), _t(self.bz[i1]))
                + self.beta * F.cross_entropy(*self._replay(i2)))
        self._step(loss)
        self._add(X, y, out.detach().cpu().numpy())
        self.t += 1


class PeriodicReplay(Learner):
    """No drift detection: every `period` batches, `steps` replay-augmented updates on the
    batches collected since the last update (isolates the value of drift triggering)."""

    def __init__(self, model, cfg, rng, scenario):
        super().__init__(model, cfg, rng, scenario)
        c = cfg["periodic"]
        self.period, self.steps = c["period"], c["steps"]
        self.mem = ClassBalancedMemory(cfg["memory_size"], scenario.X_pre.shape[1], rng)
        self.mem.add(scenario.X_pre, scenario.y_pre)
        self.buf = []

    def observe(self, X, y, probs):
        self.buf.append((X, y))
        self.t += 1
        if self.t % self.period == 0:
            Xw = np.concatenate([b[0] for b in self.buf])
            yw = np.concatenate([b[1] for b in self.buf])
            self.model.train()
            for _ in range(self.steps):
                j = self.rng.integers(0, len(yw), len(X))
                Xr, yr = self.mem.sample(len(X))
                Xt, yt = _t(np.concatenate([Xw[j], Xr]), np.concatenate([yw[j], yr]))
                self._step(F.cross_entropy(self.model(Xt), yt))
            self.mem.add(Xw, yw)
            self.buf = []
        self.extra_bytes = self.mem.nbytes() + sum(b[0].nbytes for b in self.buf)


class DriftTriggered(Learner):
    """Base for learners that only update after the hybrid detector raises an alarm."""

    uses_detector = True

    def __init__(self, model, cfg, rng, scenario):
        super().__init__(model, cfg, rng, scenario)
        self.detector = HybridDriftDetector(**cfg["detector"])
        self.adapting, self.n_adapt, self.errs = False, 0, []
        self.ref_err = 0.0

    def observe(self, X, y, probs):
        err = float((probs.argmax(1) != y).mean())  # labelled flows only
        fb = getattr(self, "full_batch", None)       # whole batch for unsupervised signals
        X_det, p_det = fb if fb is not None else (X, probs)
        ent = float(-(p_det * np.log(p_det + 1e-12)).sum(1).mean())
        if not self.adapting:
            alarm, sev = self.detector.update(X_det, err, ent)
            signals = list(self.detector.last_fired)
            if alarm:
                self.n_residual = 0
                self.__dict__.get("stable_errs", []).clear()
            elif self._residual(err):
                # unrecovered: error stays well above the best stable level although the
                # (re-baselined) detector is quiet -> resume adaptation
                alarm, sev, signals = True, 0.5, ["residual"]
                self.n_residual = getattr(self, "n_residual", 0) + 1
            if alarm:
                self.alarms.append(self.t)
                self.alarm_info.append({"t": self.t, "severity": sev, "signals": signals})
                # recovery target: best stable error level demonstrated so far
                ce = self.detector.cusum.get("error")
                base = ce.mu if ce is not None and ce.mu is not None else err
                self.ref_err = min(getattr(self, "ref_err_best", base), base)
                self.ref_err_best = self.ref_err
                self.adapting, self.n_adapt, self.errs = True, 0, []
                self.on_alarm(sev)
        if self.adapting:
            self.adapt(X, y)
            self.n_adapt += 1
            self.errs.append(err)
            if self.should_stop():
                self.adapting = False
                self.alarm_info[-1]["adapt_batches"] = self.n_adapt
                self.end_adapt()
                self.detector.reset()
        else:
            self.stable(X, y)
        self.t += 1

    def should_stop(self):
        return self.n_adapt >= self.cfg["adapt_batches"]

    def _residual(self, err):
        """Residual trigger (off unless `residual` is configured for the learner)."""
        r = self.cfg.get("residual")
        if not r or not hasattr(self, "ref_err_best"):
            return False
        hist = self.__dict__.setdefault("stable_errs", [])
        hist.append(err)
        if len(hist) < r["window"] or getattr(self, "n_residual", 0) >= r["max_consecutive"]:
            return False
        if np.mean(hist[-r["window"]:]) > self.ref_err_best + r["margin"]:
            hist.clear()
            return True
        return False

    def on_alarm(self, severity):
        pass

    def adapt(self, X, y):
        pass

    def end_adapt(self):
        pass

    def stable(self, X, y):
        pass


class FullRetrain(DriftTriggered):
    """On drift: collect an adaptation window, then retrain from scratch on all data ever seen."""

    def __init__(self, model, cfg, rng, scenario):
        super().__init__(model, cfg, rng, scenario)
        self.X_all, self.y_all = [scenario.X_pre], [scenario.y_pre]
        self.extra_bytes = scenario.X_pre.nbytes

    def _store(self, X, y):
        self.X_all.append(X)
        self.y_all.append(y)
        self.extra_bytes += X.nbytes

    def adapt(self, X, y):
        self._store(X, y)

    def stable(self, X, y):
        self._store(X, y)

    def end_adapt(self):
        reset_parameters(self.model)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg["lr"])
        self.grad_steps += train_epochs(self.model, self.opt, np.concatenate(self.X_all),
                                        np.concatenate(self.y_all),
                                        self.cfg["full_retrain"]["epochs"],
                                        self.cfg["train_batch"], self.rng)


class DACL(DriftTriggered):
    """Proposed: Drift-Aware Continual Learning.

    Stable:  inference only; labelled samples feed a class-balanced memory (no gradient cost).
    Drift:   for `adapt_batches` batches, k gradient steps per batch on
               CE(new batch) + lambda_r * CE(replay) + lambda_kd * T^2 * KL(teacher || student)
             where the teacher is a frozen snapshot taken at the alarm and KD is applied to
             replayed samples of classes known before the alarm (retain old knowledge).
             k = round(kmin + (kmax - kmin) * severity) (severity-scaled adaptation extent).
    Stop:    recovery-based termination - adaptation ends once the prequential error of the
             last 3 batches is within `recovery_tol` of the pre-drift error, or has not improved
             for `patience` batches (plateau), after at least `adapt_batches` and at most
             `max_adapt_batches` batches.
    Flags allow the ablations: use_replay, use_kd, adaptive_extent.
    """

    def __init__(self, model, cfg, rng, scenario, use_replay=True, use_kd=True,
                 adaptive_extent=True, use_gate=False):
        super().__init__(model, cfg, rng, scenario)
        self.use_replay, self.use_kd, self.adaptive = use_replay, use_kd, adaptive_extent
        d = cfg["dacl"]
        self.kmin, self.kmax = d["kmin"], d["kmax"]
        self.lam_r, self.lam_kd, self.T = d["lambda_replay"], d["lambda_kd"], d["temperature"]
        self.stable_steps = d.get("stable_steps", 0)
        self.mem = ClassBalancedMemory(cfg["memory_size"], scenario.X_pre.shape[1], rng)
        self.mem.add(scenario.X_pre, scenario.y_pre)
        self.teacher, self.k = None, self.kmax
        self.extents = []
        # classes the model has actually been trained on (not merely stored in memory:
        # samples of a new attack reach memory during the detection delay)
        self.trained = set(np.unique(scenario.y_pre).tolist())
        # optional blockchain-verified update gate (opt-in; no effect on runs without it)
        self.gate, self.gate_log = None, []
        g = cfg.get("gate", {})
        if use_gate:
            from chain.gate import make_gate, model_hash
            self._hash = model_hash
            self.scenario, self.normal_idx = scenario, scenario.normal_idx
            self.quarantine, self.q_max = [], g.get("quarantine_batches", 20)
            self.gate = make_gate(g.get("backend", "local"), threshold=g["threshold"],
                                  max_recall_drop=g["max_recall_drop"],
                                  max_detect_drop=g["max_detect_drop"],
                                  max_cum_drop=g.get("max_cum_drop", 0.05),
                                  soft_drop=g.get("soft_drop", 1.0),
                                  **({"deploy_new": g.get("deploy_new", True)}
                                     if g.get("backend") == "purechain" else {}))
            self.gate.register(self._hash(self.model))
            self.gate_policy = g.get("policy", "strict")
            # shadow adaptation: inference always uses the verified (on-chain active) model;
            # the candidate trains in a shadow copy and is swapped in only after acceptance
            self.serving = copy.deepcopy(self.model)
            self.cand_errs = []
            self.ck_every = g.get("checkpoint_every", 4)
            self.dirty, self.force_stop = False, False

    def on_alarm(self, severity):
        self.k = (int(round(self.kmin + (self.kmax - self.kmin) * severity))
                  if self.adaptive else self.kmax)
        self.extents.append(self.k)
        self.win_X, self.win_y = [], []
        if self.gate is not None:
            self.parent = {"state": copy.deepcopy(self.model.state_dict()),
                           "trained": set(self.trained), "severity": severity}
            # forgetting is checked on knowledge consolidated BEFORE this drift episode; classes
            # still being learned within the episode are covered by the detection check. A
            # residual trigger continues the same unrecovered drift, so it keeps the class set.
            residual = self.alarm_info and self.alarm_info[-1]["signals"] == ["residual"]
            if not (residual and hasattr(self, "episode_old")):
                self.episode_old = sorted(self.trained)
            # quarantined (unverified) samples join the adaptation window, so validators judge
            # their effect; they enter long-term memory only if the update is accepted
            self.win_X = [q[0] for q in self.quarantine]
            self.win_y = [q[1] for q in self.quarantine]
            self.quarantine = []
            self.cand_errs = []
            self.since_ck, self.dirty, self.force_stop = 0, False, False
        if self.use_kd:
            self.teacher = copy.deepcopy(self.model).eval()
            self.old_classes = torch.tensor(sorted(self.trained), device=dev.DEVICE)

    def _update(self, X, y, steps, window=None):
        self.model.train()
        Xt, yt = _t(X, y)
        for _ in range(steps):
            if window is not None and len(window[1]) > len(y):
                # current batch + a sample of the post-drift adaptation window
                j = self.rng.integers(0, len(window[1]), len(y))
                Xn, yn = _t(np.concatenate([X, window[0][j]]), np.concatenate([y, window[1][j]]))
            else:
                Xn, yn = Xt, yt
            loss = F.cross_entropy(self.model(Xn), yn)
            if self.use_replay or self.use_kd:
                if self.use_replay:
                    Xr, yr = _t(*self.mem.sample(len(X)))
                    out_r = self.model(Xr)
                    loss = loss + self.lam_r * F.cross_entropy(out_r, yr)
                else:
                    Xr, yr, out_r = Xt, yt, None
                if self.teacher is not None:
                    # distil only old-class logits on old-class samples, so new-class
                    # logits stay free to grow (class-incremental LwF/iCaRL style)
                    # selective: only where the teacher is still correct (stale knowledge
                    # under covariate shift is not preserved)
                    with torch.no_grad():
                        t_all = self.teacher(Xr)
                    keep = torch.isin(yr, self.old_classes) & (t_all.argmax(1) == yr)
                    if keep.any():
                        s = (out_r if out_r is not None else self.model(Xr))[keep][:, self.old_classes]
                        tlog = t_all[keep][:, self.old_classes]
                        kd = F.kl_div(F.log_softmax(s / self.T, 1), F.softmax(tlog / self.T, 1),
                                      reduction="batchmean") * self.T ** 2
                        loss = loss + self.lam_kd * kd
            self._step(loss)

    @torch.no_grad()
    def predict_proba(self, X):
        model = self.serving if self.gate is not None else self.model
        model.eval()
        return F.softmax(model(_t(X)), 1).cpu().numpy()

    def adapt(self, X, y):
        if self.gate is not None:  # prequential error of the shadow candidate (before training on X)
            self.model.eval()
            with torch.no_grad():
                self.cand_errs.append(float((self.model(_t(X)).argmax(1).cpu().numpy() != y).mean()))
        self.win_X.append(X)
        self.win_y.append(y)
        self._update(X, y, self.k, (np.concatenate(self.win_X), np.concatenate(self.win_y)))
        self._remember(X, y)
        self.trained |= set(np.unique(y).tolist())
        if self.gate is not None:
            # progressive verification: checkpoint the candidate every `ck_every` batches
            self.dirty, self.since_ck = True, self.since_ck + 1
            if self.since_ck >= self.ck_every:
                self.since_ck = 0
                if not self._gate_check():
                    self.force_stop = True  # rejected: suspected poisoning -> stop adapting

    def should_stop(self):
        d = self.cfg["dacl"]
        n = self.n_adapt
        errs = self.cand_errs if self.gate is not None else self.errs
        if self.gate is not None and self.force_stop:
            return True
        if n >= d["max_adapt_batches"]:
            return True
        if n < d.get("min_adapt_batches", self.cfg["adapt_batches"]):
            return False
        recent = np.mean(errs[-3:])
        if recent <= self.ref_err + d["recovery_tol"]:
            return True
        smooth = [np.mean(errs[max(0, i - 2):i + 1]) for i in range(n)]
        best = int(np.argmin(smooth))
        return n - 1 - best >= d["patience"]

    def end_adapt(self):
        if self.gate is not None and self.dirty:
            self._gate_check()
        self.teacher = None
        self.win_X, self.win_y = [], []

    def _gate_metrics(self, model, X, y, old):
        """Per-class detection consistency over known classes, and balanced detection accuracy.

        For a known attack class c: P(pred != Normal | c) -- the class stays *detectable*;
        for Normal: P(pred == Normal | Normal). Confusing one attack type with another still
        raises an alarm, whereas attack -> Normal is a missed attack, so the gate protects
        detectability (exact-class recall would reject benign boundary moves between attack types).
        """
        model.eval()
        with torch.no_grad():
            pred = model(_t(X)).argmax(1).cpu().numpy()
        nrm = self.normal_idx
        rec = {c: float(np.mean((pred[y == c] == nrm) if c == nrm else (pred[y == c] != nrm)))
               for c in old if (y == c).any()}
        if self.gate_policy == "net_bal":
            # balanced net detectability: half false alarms (Normal specificity), half missed
            # attacks (mean detectability over every attack class the validator holds)
            atk_cls = [c for c in np.unique(y) if c != nrm]
            spec = float(np.mean(pred[y == nrm] == nrm)) if (y == nrm).any() else 1.0
            atk = float(np.mean([np.mean(pred[y == c] != nrm) for c in atk_cls])) if atk_cls else 1.0
            return rec, 0.5 * spec + 0.5 * atk
        if self.gate_policy == "net":
            # macro detectability over ALL classes the validator holds (incl. new attacks):
            # learning a new attack raises it, unlearning a known attack lowers it
            det = [float(np.mean((pred[y == c] == nrm) if c == nrm else (pred[y == c] != nrm)))
                   for c in np.unique(y)]
            return rec, float(np.mean(det))
        atk = y != nrm
        tpr = np.mean(pred[atk] != nrm) if atk.any() else 1.0
        tnr = np.mean(pred[~atk] == nrm) if (~atk).any() else 1.0
        return rec, float((tpr + tnr) / 2)

    def _gate_check(self):
        from chain.gate import ACCEPTED, PENDING
        parent_model = copy.deepcopy(self.model)
        parent_model.load_state_dict(self.parent["state"])
        old = self.episode_old
        uid = self.gate.propose(self._hash(self.model), len(self.alarms), self.parent["severity"])
        status, votes = PENDING, []
        for v, (Xv, yv) in enumerate(self.scenario.validator_sets(self.phase)):
            c_rec, c_det = self._gate_metrics(self.model, Xv, yv, old)
            p_rec, p_det = self._gate_metrics(parent_model, Xv, yv, old)
            # report the worst-affected known class: losing detectability of a single attack
            # class must not be hidden by averaging over classes
            worst = max(c_rec, key=lambda c: p_rec[c] - c_rec[c]) if c_rec else None
            cand = (c_rec[worst] if worst is not None else 1.0, c_det)
            par = (p_rec[worst] if worst is not None else 1.0, p_det)
            votes.append({"cand": cand, "parent": par, "worst_class": worst})
            status = self.gate.attest(v, uid, worst if worst is not None else 0, cand, par)
            if status != PENDING:
                break
        accepted = status == ACCEPTED
        self.gate_log.append({"t": self.t, "update": int(uid), "accepted": accepted, "votes": votes,
                              "escalated": int(uid) in self.gate.escalated})
        if accepted:  # promote the candidate; verified knowledge enters long-term replay memory
            self.serving.load_state_dict(self.model.state_dict())
            self.mem.add(np.concatenate(self.win_X), np.concatenate(self.win_y))
            # the promoted model is the parent of the next checkpoint
            self.parent["state"] = copy.deepcopy(self.model.state_dict())
            self.parent["trained"] = set(self.trained)
        else:  # roll back model and learned classes; discard the unverified window
            self.model.load_state_dict(self.parent["state"])
            self.trained = set(self.parent["trained"])
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg["lr"])
        self.win_X, self.win_y, self.dirty = [], [], False
        return accepted

    def stable(self, X, y):
        if self.stable_steps:
            self._update(X, y, self.stable_steps)
        self._remember(X, y)

    def _remember(self, X, y):
        if self.gate is not None:  # memory only changes through accepted updates
            if not self.adapting:
                self.quarantine = (self.quarantine + [(X, y)])[-self.q_max:]
        else:
            self.mem.add(X, y)
        self.extra_bytes = (self.mem.nbytes() + (param_bytes(self.model) if self.teacher else 0)
                            + (param_bytes(self.model) if self.gate is not None else 0)
                            + sum(w.nbytes for w in getattr(self, "win_X", []))
                            + sum(q[0].nbytes for q in getattr(self, "quarantine", [])))


class WindowRetrain(DriftTriggered):
    """Drift-adaptive sliding-window IDS (as in drift-adaptive IoT IDS work, e.g. Yang & Shami 2021):
    on a drift alarm, collect `adapt_batches` batches, then retrain from scratch on the most recent
    `window_batches` labelled batches only (no long-term memory). Same detector as DACL, so only the
    adaptation strategy differs."""

    def __init__(self, model, cfg, rng, scenario):
        super().__init__(model, cfg, rng, scenario)
        w = cfg.get("window_retrain", {"window_batches": 20, "epochs": 5})
        self.window, self.epochs = w["window_batches"], w["epochs"]
        self.recent = []

    def _store(self, X, y):
        self.recent = (self.recent + [(X, y)])[-self.window:]
        self.extra_bytes = sum(x.nbytes for x, _ in self.recent)

    def adapt(self, X, y):
        self._store(X, y)

    def stable(self, X, y):
        self._store(X, y)

    def end_adapt(self):
        Xw = np.concatenate([x for x, _ in self.recent])
        yw = np.concatenate([y for _, y in self.recent])
        reset_parameters(self.model)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg["lr"])
        self.grad_steps += train_epochs(self.model, self.opt, Xw, yw, self.epochs,
                                        self.cfg["train_batch"], self.rng)


class DACLGateDual(DACL):
    """Dual serving: the *security decision* (alarm vs. normal) comes only from the PureChain-verified
    model; the *attack-type label* of an alarm comes from an advisory DACL that keeps learning without
    the gate. The advisor can relabel an alarm but never suppress one, so a poisoned advisor can at
    worst mis-attribute an attack type."""

    def __init__(self, model, cfg, rng, scenario):
        advisor_model = copy.deepcopy(model)
        super().__init__(model, cfg, rng, scenario, use_gate=True)
        self.advisor = DACL(advisor_model, cfg, rng, scenario)

    def _verified_proba(self, X):
        return DACL.predict_proba(self, X)

    @torch.no_grad()
    def predict_proba(self, X):
        p_ver = self._verified_proba(X)
        p_adv = self.advisor.predict_proba(X)
        n = self.normal_idx
        alarm = p_ver.argmax(1) != n
        q = p_ver.copy()
        a = p_adv[alarm].copy()
        a[:, n] = 0.0
        s = a.sum(1, keepdims=True)
        ok = s[:, 0] > 1e-9
        rows = np.flatnonzero(alarm)
        q[rows[ok]] = a[ok] / s[ok]  # advisor's attack type for verified alarms
        return q

    def observe(self, X, y, probs):
        before = self.advisor.grad_steps
        self.advisor.observe(X, y, self.advisor.predict_proba(X))
        DACL.observe(self, X, y, self._verified_proba(X))
        self.grad_steps += self.advisor.grad_steps - before  # both models' training counts as cost

    def state_bytes(self):
        return super().state_bytes() + self.advisor.state_bytes()


METHODS = {
    "Static": lambda m, c, r, s: Static(m, c, r, s),
    "FineTune": lambda m, c, r, s: FineTune(m, c, r, s),
    "ER": lambda m, c, r, s: ExperienceReplay(m, c, r, s),
    "EWC": lambda m, c, r, s: OnlineEWC(m, c, r, s),
    "FullRetrain": lambda m, c, r, s: FullRetrain(m, c, r, s),
    "DriftFT": lambda m, c, r, s: DACL(m, c, r, s, use_replay=False, use_kd=False),
    "DACL": lambda m, c, r, s: DACL(m, c, r, s),
    "DACL-noKD": lambda m, c, r, s: DACL(m, c, r, s, use_kd=False),
    "DACL-noReplay": lambda m, c, r, s: DACL(m, c, r, s, use_replay=False),
    "DACL-fixedK": lambda m, c, r, s: DACL(m, c, r, s, adaptive_extent=False),
    "LwF": lambda m, c, r, s: LwF(m, c, r, s),
    "DER++": lambda m, c, r, s: DERpp(m, c, r, s),
    "PeriodicER": lambda m, c, r, s: PeriodicReplay(m, c, r, s),
    "DACL+Gate": lambda m, c, r, s: DACL(m, c, r, s, use_gate=True),
    "DACL+Gate-dual": lambda m, c, r, s: DACLGateDual(m, c, r, s),
    "WindowRetrain": lambda m, c, r, s: WindowRetrain(m, c, r, s),
}
