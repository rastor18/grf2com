"""Trials, feature building (training-subject normalization), splits, and the on-device window sampler."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import preprocess as pp
from .anthro import SubjectHeight
from .tables import (check_subject_features, development_subjects, final_test_subjects, load_infotable,
                     load_trial_index)

CH = {c: i for i, c in enumerate(pp.CHANNELS)}
GROUPS = {
    "forces": ["Fx_l", "Fx_r", "Fy_l", "Fy_r", "Fz_l", "Fz_r"],
    "cop": ["COPx_l", "COPx_r", "COPy_l", "COPy_r"],
    "moments": ["Mx_l", "Mx_r", "My_l", "My_r", "Mz_l", "Mz_r"],
    "contact": ["contact_l", "contact_r"],
    "speed": ["speed"],
    "total_force": ["Fx_tot", "Fy_tot", "Fz_tot"],   # derived
    "net_cop": ["netCOPx", "netCOPy"],             # derived: Fz-weighted, per-plate COP centred on training means
    # Single-belt (one plate under both feet), derived from the two plates in a common frame (see _combined_cop):
    "single_cop": ["COPx_c", "COPy_c"],
    "single_moments": ["Mx_c", "My_c"],            # Mx = Fz COPy, My = -Fz COPx about the common origin
    # State-estimator outputs, run causally over the whole trial (memory beyond the network's window):
    "lab_kf": ["labkf_p", "labkf_v", "labkf_since_step"],   # lab Kalman filter (baselines/song_kf), fold-independent
    "kalman_state": ["kal_ap", "kal_ml", "kal_vert"],        # our force-integration Kalman, fit on training subjects
}
PER_BW = {"forces", "moments", "total_force", "single_moments"}   # divided by body weight when data.force_per_bw
SPLIT_BELT_GROUPS = {"forces", "cop", "moments", "contact", "net_cop"}


def is_single_belt(cfg: dict) -> bool:
    """True when no per-plate input is used, i.e. the inputs a regular (single-belt) instrumented treadmill gives."""
    return not (set(cfg["data"]["features"]) & SPLIT_BELT_GROUPS)


def rate_mult(cfg: dict) -> int:
    """Input samples per CoM frame."""
    return int(cfg["preprocess"]["grf_rate"]) // pp.COM_RATE


@dataclass
class Trial:
    subject: str
    trial: str
    x: np.ndarray          # (n * k, len(pp.CHANNELS)) cached channels
    y: np.ndarray          # (n, 3) CoM in m, NaN where missing
    mask: np.ndarray       # (n,) valid target
    events: dict = field(default_factory=dict)   # HS/TO frame indices (evaluation only)
    mass: float = float("nan")

    @property
    def name(self) -> str:
        return f"{self.subject}_{self.trial}"

    @property
    def n_frames(self) -> int:
        return len(self.y)


def load_trials(cfg: dict, pairs, cdir: Path | None = None) -> list[Trial]:
    cdir = cdir or pp.cache_dir(cfg)
    info = load_infotable(cfg)
    delay = int(round(cfg["target"].get("delay_ms", 0) * pp.COM_RATE / 1000))
    out = []
    for s, t in pairs:
        d = pp.load_trial(cdir, s, t)
        y, mask = d["y"], d["mask"]
        if delay:   # predict CoM at t - delay
            y = np.concatenate([np.full((delay, 3), np.nan, np.float32), y[:-delay]])
            mask = np.concatenate([np.zeros(delay, bool), mask[:-delay]])
        ev = {k[3:]: v for k, v in d.items() if k.startswith("ev_")}
        out.append(Trial(s, t, d["x"], y, mask, ev, float(info.loc[s, "mass"])))
    return out


VERTICAL_MODES = ("absolute", "anthro_residual", "anthro_centered")


def combined_cop(x: np.ndarray, belt_offset_mm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """COP (mm) and Fz (N) a single plate under both belts would measure: the Fz-weighted mean of the per-plate COP in
    a common frame. Each plate's ML coordinate is shifted by +/- the belt width (left +, right -), as the lab's
    treadmill_controller.read_bertec does; AP is already shared. Unloaded plates (or NaN COP) get zero weight."""
    w = {s: np.clip(x[:, CH[f"Fz_{s}"]], 0, None) for s in "lr"}
    shift = {"l": belt_offset_mm, "r": -belt_offset_mm}
    num_x, num_y = 0.0, 0.0
    for s in "lr":
        cx, cy = x[:, CH[f"COPx_{s}"]] + shift[s], x[:, CH[f"COPy_{s}"]]
        ok = np.isfinite(cx) & np.isfinite(cy)
        w[s] = np.where(ok, w[s], 0.0)
        num_x = num_x + w[s] * np.nan_to_num(cx)
        num_y = num_y + w[s] * np.nan_to_num(cy)
    total = w["l"] + w["r"]
    den = np.where(total > 1e-6, total, np.nan)
    return num_x / den, num_y / den, x[:, CH["Fz_l"]] + x[:, CH["Fz_r"]]


class FeatureBuilder:
    """Builds model inputs and targets from cached channels. All statistics come from the training trials passed to
    fit().

    target.vertical sets what the network learns for vertical CoM:
      absolute        - the absolute height
      anthro_residual - height minus the anthropometric estimate (regression on training subjects)
      anthro_centered - height minus each training subject's own mean height, so the network cannot learn per-subject
                        offsets; val / test subjects use the anthropometric estimate
    In both anthro modes predictions are returned as absolute height: network output + the anthropometric estimate.
    """

    def __init__(self, cfg: dict):
        d = cfg["data"]
        self.cfg = cfg
        self.groups = list(d["features"])
        unknown = [g for g in self.groups if g not in GROUPS]
        if unknown:
            raise ValueError(f"unknown feature group(s) {unknown}; choose from {list(GROUPS)}")
        self.per_bw = bool(d.get("force_per_bw", True))
        self.belt_offset = float(d.get("belt_ml_offset_mm", 508.0))
        self.subject_features = check_subject_features(d.get("subject_features") or [])
        self.names = [c for g in self.groups for c in GROUPS[g]]
        self.vertical = (cfg.get("target") or {}).get("vertical", "absolute")
        if self.vertical not in VERTICAL_MODES:
            raise ValueError(f"target.vertical must be one of {VERTICAL_MODES}, got {self.vertical!r}")
        self.state: dict = {}
        self._height: SubjectHeight | None = None
        self._info: pd.DataFrame | None = None
        self._kalman = None
        self._est_cache: dict = {}

    @property
    def info(self) -> pd.DataFrame:
        if self._info is None:
            self._info = load_infotable(self.cfg)
        return self._info

    def vertical_offset(self, subject: str, for_target: bool = False) -> float:
        """Metres subtracted from the vertical target and added back to predictions (0 in absolute mode)."""
        if self.vertical == "absolute":
            return 0.0
        if for_target and self.vertical == "anthro_centered" and subject in self._height.subject_means:
            return self._height.subject_means[subject]
        return self._height.predict(subject, self.info)

    def _offset_y(self, t: Trial) -> np.ndarray:
        y = t.y.astype(np.float64)
        if self.vertical != "absolute":
            y = y.copy()
            y[:, 2] -= self.vertical_offset(t.subject, for_target=True)
        return y

    # -- raw (unnormalized) features ------------------------------------------------------------
    def _raw(self, t: Trial, cop_center: dict | None = None) -> np.ndarray:
        x = t.x
        bw = t.mass * pp.G
        cols = []
        for g in self.groups:
            if g == "total_force":
                block = np.stack([x[:, CH[f"F{a}_l"]] + x[:, CH[f"F{a}_r"]] for a in "xyz"], 1)
            elif g == "net_cop":
                block = self._net_cop(x, cop_center if cop_center is not None else self.state["cop_center"])
            elif g in ("lab_kf", "kalman_state"):
                block = np.repeat(self._estimator_block(t, g), len(x) // t.n_frames, axis=0)[:len(x)]
            elif g in ("single_cop", "single_moments"):
                cx, cy, fz = combined_cop(x, self.belt_offset)
                block = np.stack([cx, cy], 1) if g == "single_cop" else np.stack([fz * cy, -fz * cx], 1) / 1000
            else:
                block = x[:, [CH[c] for c in GROUPS[g]]]
            if self.per_bw and g in PER_BW:
                block = block / bw
            cols.append(block.astype(np.float64))
        return np.concatenate(cols, axis=1)

    @staticmethod
    def _net_cop(x: np.ndarray, center: dict) -> np.ndarray:
        wl = np.clip(x[:, CH["Fz_l"]], 0, None)
        wr = np.clip(x[:, CH["Fz_r"]], 0, None)
        out = []
        for a in "xy":
            cl = np.nan_to_num(x[:, CH[f"COP{a}_l"]] - center[f"COP{a}_l"])
            cr = np.nan_to_num(x[:, CH[f"COP{a}_r"]] - center[f"COP{a}_r"])
            out.append((wl * cl + wr * cr) / (wl + wr + 1e-6))
        return np.stack(out, 1)

    # -- fitting ----------------------------------------------------------------------------------
    def _estimator_block(self, t: Trial, group: str) -> np.ndarray:
        """Per-frame (n, 3) outputs of a state estimator for a trial, cached per trial."""
        key = (group, t.name)
        if key not in self._est_cache:
            if group == "lab_kf":
                from .baselines import song_kf
                e = song_kf.estimate(self.cfg, t)
                self._est_cache[key] = np.stack([e["p"], e["v"], e["since_step"]], 1)
            else:
                self._est_cache[key] = self._kalman.predict(t, self.info)
        return self._est_cache[key]

    def fit(self, trials: list[Trial], infotable: pd.DataFrame) -> "FeatureBuilder":
        self._info = infotable
        if "kalman_state" in self.groups:   # the Kalman's few parameters come from the training subjects only
            from .baselines.kalman import KalmanBaseline
            self._kalman = KalmanBaseline(self.cfg).fit(trials, [], infotable)
        if self.vertical != "absolute":
            self._height = SubjectHeight().fit(trials, infotable)
        cop_center = {}
        for c in GROUPS["cop"]:
            side = c[-1]
            vals = np.concatenate([t.x[t.x[:, CH[f"contact_{side}"]] > 0.95, CH[c]] for t in trials])
            cop_center[c] = float(np.nanmean(vals))
        n = np.zeros(len(self.names))
        s1 = np.zeros(len(self.names))
        s2 = np.zeros(len(self.names))
        ys, yn = [], 0
        for t in trials:
            r = self._raw(t, cop_center)
            ok = np.isfinite(r)
            n += ok.sum(0)
            s1 += np.where(ok, r, 0).sum(0)
            s2 += np.where(ok, r * r, 0).sum(0)
            ys.append(self._offset_y(t)[t.mask])
        mean = s1 / np.maximum(n, 1)
        std = np.sqrt(np.maximum(s2 / np.maximum(n, 1) - mean ** 2, 0))
        std[std < 1e-8] = 1.0
        yall = np.concatenate(ys)
        subjects = sorted({t.subject for t in trials})
        sv = infotable.loc[subjects, self.subject_features].to_numpy(float) if self.subject_features else None
        self.state = {
            "names": self.names, "cop_center": cop_center, "mean": mean.tolist(), "std": std.tolist(),
            "y_mean": yall.mean(0).tolist(), "y_std": yall.std(0).tolist(),
            "subject_features": self.subject_features, "train_subjects": subjects,
            "s_mean": sv.mean(0).tolist() if sv is not None else [],
            "s_std": np.where(sv.std(0) > 1e-8, sv.std(0), 1.0).tolist() if sv is not None else [],
            "vertical": self.vertical,
            "height": self._height.state() if self._height is not None else None,
            "kalman": self._kalman,
        }
        return self

    def load_state(self, state: dict) -> "FeatureBuilder":
        if state["names"] != self.names:
            raise ValueError("feature names in the checkpoint do not match the config")
        if state.get("vertical", "absolute") != self.vertical:
            raise ValueError("target.vertical in the checkpoint does not match the config")
        self.state = state
        self._height = SubjectHeight.from_state(state["height"]) if state.get("height") else None
        self._kalman = state.get("kalman")
        return self

    # -- transforms -------------------------------------------------------------------------------
    def transform(self, t: Trial) -> np.ndarray:
        r = (self._raw(t) - np.asarray(self.state["mean"])) / np.asarray(self.state["std"])
        return np.nan_to_num(r, nan=0.0).astype(np.float32)

    def subject_vector(self, subject: str, infotable: pd.DataFrame) -> np.ndarray:
        if not self.subject_features:
            return np.zeros(0, np.float32)
        v = infotable.loc[subject, self.subject_features].to_numpy(float)
        return ((v - np.asarray(self.state["s_mean"])) / np.asarray(self.state["s_std"])).astype(np.float32)

    def target(self, t: Trial) -> np.ndarray:
        y = (self._offset_y(t) - np.asarray(self.state["y_mean"])) / np.asarray(self.state["y_std"])
        return np.nan_to_num(y, nan=0.0).astype(np.float32)

    def denormalize(self, y: np.ndarray, subject: str | None = None) -> np.ndarray:
        """Network output -> CoM in m. Anthro vertical modes need the subject to add the height estimate back."""
        out = y * np.asarray(self.state["y_std"]) + np.asarray(self.state["y_mean"])
        if self.vertical != "absolute":
            if subject is None:
                raise ValueError(f"target.vertical={self.vertical} needs the subject to denormalize")
            out[:, 2] += self.vertical_offset(subject)
        return out

    @property
    def n_inputs(self) -> int:
        return len(self.names)

    @property
    def n_subject(self) -> int:
        return len(self.subject_features)


# -- splits ---------------------------------------------------------------------------------------
@dataclass
class Fold:
    name: str
    train: list[tuple[str, str]]
    val: list[tuple[str, str]]
    test: list[tuple[str, str]]

    def subjects(self, part: str) -> list[str]:
        return sorted({s for s, _ in getattr(self, part)})


def _trials_by_subject(cfg: dict) -> dict[str, list[str]]:
    index = load_trial_index(cfg)
    out: dict[str, list[str]] = {}
    for s, t in zip(index.subject, index.trial):
        out.setdefault(s, []).append(t)
    return {s: sorted(ts) for s, ts in out.items()}


def _check_no_locked(cfg: dict, folds: list[Fold]) -> None:
    locked = set(final_test_subjects(cfg))
    for f in folds:
        for part in ("train", "val", "test"):
            bad = sorted({s for s, _ in getattr(f, part)} & locked)
            if bad:
                raise RuntimeError(f"locked final-test subject(s) {bad} in development fold {f.name} ({part})")


def make_folds(cfg: dict, dev_mode: bool = False, only: list[str] | None = None) -> list[Fold]:
    """Development folds over the non-locked subjects. Locked final-test subjects can never appear."""
    by_subject = _trials_by_subject(cfg)
    dev = development_subjects(cfg, by_subject)
    if dev_mode:
        k = int(cfg["data"]["dev_trials_per_subject"])
        by_subject = {s: ts[:k] for s, ts in by_subject.items()}
    pairs = {s: [(s, t) for t in by_subject[s]] for s in dev}
    split = cfg["data"]["split"]
    folds = []
    if split == "loso":
        off = int(cfg["data"].get("val_offset", 1))
        for i, s in enumerate(dev):
            v = dev[(i + off) % len(dev)]
            train = [p for u in dev if u not in (s, v) for p in pairs[u]]
            folds.append(Fold(s, train, pairs[v], pairs[s]))
    elif split == "trial_holdout":
        th = cfg["data"]["trial_holdout"]
        val_t, test_t = set(th["val"]), set(th["test"])
        allp = [p for s in dev for p in pairs[s]]
        folds.append(Fold("trial_holdout", [p for p in allp if p[1] not in val_t | test_t],
                          [p for p in allp if p[1] in val_t], [p for p in allp if p[1] in test_t]))
    else:
        raise ValueError(f"unknown split {split!r}")
    if only:
        names = {f.name for f in folds}
        missing = sorted(set(only) - names)
        if missing:
            raise ValueError(f"unknown fold(s) {missing}; available: {sorted(names)}")
        folds = [f for f in folds if f.name in only]
    _check_no_locked(cfg, folds)
    return folds


def final_fold(cfg: dict) -> Fold:
    """Train on every development trial, test on the locked subjects. Only scripts/final_test.py calls this."""
    by_subject = _trials_by_subject(cfg)
    dev = development_subjects(cfg, by_subject)
    locked = final_test_subjects(cfg)
    if not locked:
        raise RuntimeError("no final_test_subjects configured")
    return Fold("final", [(s, t) for s in dev for t in by_subject[s]], [],
                [(s, t) for s in locked for t in by_subject.get(s, [])])


# -- tensors on the device and window sampling -----------------------------------------------------
class TensorSet:
    """All trials of one split concatenated on the device; training windows are gathered by index."""

    def __init__(self, trials: list[Trial], fb: FeatureBuilder, infotable: pd.DataFrame, k: int,
                 device: torch.device):
        self.trials, self.k, self.device = trials, k, device
        self.lengths = np.array([t.n_frames for t in trials])
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)[:-1]])
        self.x = torch.from_numpy(np.concatenate([fb.transform(t) for t in trials])).to(device)
        self.y = torch.from_numpy(np.concatenate([fb.target(t) for t in trials])).to(device)
        self.m = torch.from_numpy(np.concatenate([t.mask for t in trials])).to(device)
        self.s = torch.from_numpy(np.stack([fb.subject_vector(t.subject, infotable) for t in trials])).to(device)

    def features(self, i: int) -> torch.Tensor:
        """Inputs of trial i, (n * k, C), on the device."""
        a = self.offsets[i] * self.k
        return self.x[a:a + self.lengths[i] * self.k]

    def sample(self, batch: int, length: int, warmup: int, rng: np.random.Generator):
        """Random windows of `length` frames. Returns x (B, L*k, C), s (B, S), y (B, L, 3), loss mask (B, L)."""
        room = np.maximum(self.lengths - length + 1, 0)
        ti = rng.choice(len(self.trials), size=batch, p=room / room.sum())
        start = self.offsets[ti] + (rng.random(batch) * room[ti]).astype(np.int64)
        dev = self.device
        frames = torch.from_numpy(start)[:, None].to(dev) + torch.arange(length, device=dev)[None]
        inputs = (torch.from_numpy(start * self.k)[:, None].to(dev)
                  + torch.arange(length * self.k, device=dev)[None])
        loss_mask = self.m[frames] & (torch.arange(length, device=dev)[None] >= warmup)
        return self.x[inputs], self.s[torch.from_numpy(ti).to(dev)], self.y[frames], loss_mask
