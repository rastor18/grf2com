"""Per-trial preprocessing into cache/<hash>/<subject>_<trial>.npz.

Steps: zero each plate (tare from its unloaded periods), contact flag, hold COP while unloaded, causal Butterworth
low-pass, decimate to preprocess.grf_rate, align to the 60 Hz CoM frames, resample the belt-speed log with a causal
hold, and build the valid-target mask. Everything applied sample by sample is causal; the tare is one constant per
trial (see tare_window_s).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, signal

from .config import config_hash, project_path
from .tables import load_events, load_infotable, load_trial_index

PREPROCESS_VERSION = 1   # bump to invalidate caches after a change to this file
GRF_RATE = 1200
COM_RATE = 60
FRAME_STEP = GRF_RATE // COM_RATE   # GRF rows per CoM frame
G = 9.81
SIDES = {"l": "left", "r": "right"}
TARED = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
CHANNELS = [f"{q}_{s}" for q in ("Fx", "Fy", "Fz", "COPx", "COPy", "Mx", "My", "Mz") for s in SIDES]
CHANNELS += ["contact_l", "contact_r", "speed"]
TARGETS = ["CoM_Pos_1", "CoM_Pos_2", "CoM_Pos_3"]
EVENTS = ["HS_left", "HS_right", "TO_left", "TO_right"]


def hold_last(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Replace samples where `valid` is False with the last valid sample (NaN before the first one). Causal."""
    idx = np.where(valid, np.arange(len(x)), -1)
    np.maximum.accumulate(idx, out=idx)
    return np.where(idx >= 0, x[np.maximum(idx, 0)], np.nan)


def causal_filter(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Causal SOS filter started in steady state at the first finite sample; NaN before it. NaNs after it are held."""
    finite = np.isfinite(x)
    out = np.full(len(x), np.nan)
    if not finite.any():
        return out
    i0 = int(np.argmax(finite))
    xs = hold_last(x, finite)[i0:]
    out[i0:], _ = signal.sosfilt(sos, xs, zi=signal.sosfilt_zi(sos) * xs[0])
    return out


def unloaded_mask(fz: np.ndarray, margin_n: float, erode: int) -> tuple[np.ndarray, int]:
    """Samples where a plate is unloaded: raw Fz below a low percentile + margin, eroded away from transitions."""
    structure = np.ones(2 * erode + 1, bool) if erode else None
    for pct in (1, 5):
        mask = fz < np.nanpercentile(fz, pct) + margin_n
        if structure is not None:
            mask = ndimage.binary_erosion(mask, structure=structure)
        if mask.sum() >= GRF_RATE:   # at least 1 s
            break
    return mask, pct


def _median(x: np.ndarray, mask: np.ndarray) -> float:
    return float(np.nanmedian(x[mask])) if mask.any() else float("nan")


def process_trial(subject: str, trial: str, pcfg: dict, data_dir: Path, mass_kg: float,
                  events: pd.DataFrame | None = None) -> tuple[dict, dict]:
    """Returns (arrays for the npz, summary row)."""
    rate = int(pcfg["grf_rate"])
    if GRF_RATE % rate or rate % COM_RATE:
        raise ValueError(f"grf_rate must divide {GRF_RATE} and be a multiple of {COM_RATE}, got {rate}")
    step, k = GRF_RATE // rate, rate // COM_RATE

    stem = data_dir / f"{subject}_{trial}"
    grf = pd.read_csv(f"{stem}_GRF.csv", engine="pyarrow")
    com = pd.read_csv(f"{stem}_CoM.csv", engine="pyarrow")
    tm = pd.read_csv(f"{stem}_Treadmill.csv", engine="pyarrow")

    t = grf["time"].to_numpy(float)
    n_grf, n_com = len(t), len(com)
    sos = signal.butter(int(pcfg["filter_order"]), float(pcfg["lowpass_hz"]), fs=GRF_RATE, output="sos")
    erode = int(round(pcfg["unloaded_erode_ms"] * GRF_RATE / 1000))
    tare_window = pcfg.get("tare_window_s")

    full = {}   # filtered 1200 Hz channels
    info = {"subject": subject, "trial": trial, "n_grf": n_grf, "n_com": n_com}
    for s, side in SIDES.items():
        fz_raw = grf[f"Fz_{side}"].to_numpy(float)
        unloaded, pct = unloaded_mask(fz_raw, pcfg["unloaded_margin_n"], erode)
        info[f"unloaded_s_{s}"] = unloaded.sum() / GRF_RATE
        info[f"unloaded_pct_{s}"] = pct
        tare = unloaded & (t < t[0] + tare_window) if tare_window else unloaded
        first, last = unloaded & (t < t[0] + 60), unloaded & (t >= t[-1] - 60)
        for q in TARED:
            x = grf[f"{q}_{side}"].to_numpy(float)
            off = _median(x, tare)
            info[f"off_{q}_{s}"] = off
            info[f"drift_{q}_{s}"] = _median(x, last) - _median(x, first)
            full[f"{q}_{s}"] = causal_filter(x - off, sos)
        fz0 = fz_raw - info[f"off_Fz_{s}"]
        contact = fz0 > pcfg["contact_threshold_n"]
        info[f"contact_frac_{s}"] = float(contact.mean())
        full[f"contact_{s}"] = causal_filter(contact.astype(float), sos)
        if pcfg.get("cop_from_moments", True):
            # The source COP is ~1000 * M / Fz from the un-zeroed channels plus a trial-specific offset of metres,
            # so it is not in a consistent frame. Recompute it from the tared moments (N*m) and Fz, in mm.
            with np.errstate(divide="ignore", invalid="ignore"):
                cops = {"COPx": -1000 * (grf[f"My_{side}"].to_numpy(float) - info[f"off_My_{s}"]) / fz0,
                        "COPy": 1000 * (grf[f"Mx_{side}"].to_numpy(float) - info[f"off_Mx_{s}"]) / fz0}
        else:
            cops = {q: grf[f"{q}_{side}"].to_numpy(float) for q in ("COPx", "COPy")}
        for q, cop in cops.items():
            full[f"{q}_{s}"] = causal_filter(hold_last(cop, contact), sos)

    # Grid: input sample i <-> GRF row i * step; CoM frame j <-> input sample j * k <-> GRF row 20 j.
    n = min(n_com, ((n_grf - 1) // step + 1) // k)
    rows = np.arange(n * k) * step
    tt, sp = tm["time"].to_numpy(float), tm["speed"].to_numpy(float)
    if np.any(np.diff(tt) < 0):
        raise ValueError(f"{subject} {trial}: treadmill log times are not sorted")
    idx = np.searchsorted(tt, t[rows], side="right") - 1
    speed = np.where(idx >= 0, sp[np.maximum(idx, 0)], 0.0) + 0.0   # + 0.0 turns -0 into 0
    x = np.stack([full[c][rows] for c in CHANNELS[:-1]] + [speed], axis=1).astype(np.float32)

    y_all = com[TARGETS].to_numpy(float)
    j = np.arange(n) + int(pcfg["align_shift_frames"])
    y = np.full((n, 3), np.nan)
    ok = (j >= 0) & (j < n_com)
    y[ok] = y_all[j[ok]]
    frame_rows = rows[::k]
    fz_total = full["Fz_l"][frame_rows] + full["Fz_r"][frame_rows]
    bw = mass_kg * G
    mask = np.isfinite(y).all(axis=1) & (fz_total > pcfg["valid_min_bw"] * bw)

    arrays = {"x": x, "y": y.astype(np.float32), "mask": mask}
    if events is not None:
        ev = events[(events.subject == subject) & (events.trial == trial)]
        for name in EVENTS:
            si = ev.loc[ev.event == name, "sample_index"].to_numpy()
            fr = np.rint((si - 1) / FRAME_STEP).astype(np.int64)   # sample_index is probably 1-based
            arrays[f"ev_{name}"] = np.sort(fr[(fr >= 0) & (fr < n)])

    info.update({
        "n_frames": n,
        "duration_s": n / COM_RATE,
        "mass_kg": mass_kg,
        "com_nan_frac": float(np.isnan(y_all).any(axis=1).mean()),
        "valid_frac": float(mask.mean()),
        "fz_total_bw_mean": float(np.nanmean(fz_total[mask]) / bw) if mask.any() else float("nan"),
        "speed_min": float(speed.min()),
        "speed_max": float(speed.max()),
        "n_treadmill_rows": len(tm),
        "n_hs_left": int(len(arrays.get("ev_HS_left", []))),
        "n_hs_right": int(len(arrays.get("ev_HS_right", []))),
    })
    for a, name in enumerate(TARGETS):
        info[f"{name}_min"] = float(np.nanmin(y[:, a])) if np.isfinite(y[:, a]).any() else float("nan")
        info[f"{name}_max"] = float(np.nanmax(y[:, a])) if np.isfinite(y[:, a]).any() else float("nan")
    return arrays, info


def cache_key(cfg: dict) -> dict:
    pcfg = {k: v for k, v in cfg["preprocess"].items() if k != "workers"}
    return {"version": PREPROCESS_VERSION, "preprocess": pcfg, "data_dir": cfg["paths"]["data_dir"]}


def cache_dir(cfg: dict) -> Path:
    return project_path(cfg, "cache_dir") / config_hash(cache_key(cfg))


def trial_path(cdir: Path, subject: str, trial: str) -> Path:
    return cdir / f"{subject}_{trial}.npz"


def _run_one(subject, trial, pcfg, data_dir, mass_kg, events, cdir: Path) -> dict:
    t0 = time.perf_counter()
    arrays, info = process_trial(subject, trial, pcfg, data_dir, mass_kg, events)
    info["seconds"] = round(time.perf_counter() - t0, 2)
    out = trial_path(cdir, subject, trial)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, out)
    out.with_suffix(".json").write_text(json.dumps(info, indent=1))
    return info


def run(cfg: dict, trials: list[tuple[str, str]] | None = None, workers: int | None = None,
        force: bool = False, verbose: bool = True) -> Path:
    """Preprocess `trials` (default: every trial in trial_index.csv) into cache_dir(cfg); returns that directory."""
    from joblib import Parallel, delayed

    cdir = cache_dir(cfg)
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "meta.json").write_text(json.dumps({
        **cache_key(cfg), "channels": CHANNELS, "targets": TARGETS, "events": EVENTS,
        "grf_rate": cfg["preprocess"]["grf_rate"], "com_rate": COM_RATE}, indent=1))
    index = load_trial_index(cfg)
    all_trials = list(zip(index.subject, index.trial))
    trials = all_trials if trials is None else trials
    todo = [st for st in trials if force or not trial_path(cdir, *st).with_suffix(".json").exists()]
    if todo:
        info = load_infotable(cfg)
        events = load_events(cfg)
        data_dir = project_path(cfg, "data_dir")
        pcfg = cfg["preprocess"]
        workers = workers or int(pcfg.get("workers", 1))
        if verbose:
            print(f"preprocessing {len(todo)} trial(s) with {workers} worker(s) -> {cdir}", flush=True)
        jobs = (delayed(_run_one)(s, t, pcfg, data_dir, float(info.loc[s, "mass"]),
                                  events[(events.subject == s) & (events.trial == t)], cdir) for s, t in todo)
        for row in Parallel(n_jobs=workers, return_as="generator_unordered")(jobs):
            if verbose:
                print(f"  {row['subject']} {row['trial']}: {row['n_frames']} frames, "
                      f"valid {row['valid_frac']:.3f}, Fz/BW {row['fz_total_bw_mean']:.3f} ({row['seconds']} s)",
                      flush=True)
    rows = [json.loads(trial_path(cdir, s, t).with_suffix(".json").read_text())
            for s, t in all_trials if trial_path(cdir, s, t).with_suffix(".json").exists()]
    pd.DataFrame(rows).to_csv(cdir / "summary.csv", index=False)
    return cdir


def load_trial(cdir: Path, subject: str, trial: str) -> dict:
    with np.load(trial_path(cdir, subject, trial)) as z:
        return {key: z[key] for key in z.files}


def load_summary(cdir: Path) -> pd.DataFrame:
    return pd.read_csv(cdir / "summary.csv")
