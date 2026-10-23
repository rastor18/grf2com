"""Scoring on treadmill speed transitions (evaluation only; never used for training or model inputs).

The commanded speed logs have a start-up ramp, a steady 1.25 m/s stretch, then a pulse phase: 1.25 -> 0.75 -> 1.25 and
1.25 -> 1.75 -> 1.25 m/s pulses (each edge ~0.43 s, the return 3 s after the first edge, the next pulse ~7-8 s later),
then a stop. In steady walking a model can look good by reproducing the average gait cycle; pulses force it to use its
inputs (stride and forces change, the subject drifts on the belt).

Definitions (60 Hz CoM frames; frame j is at j / 60 s on the force-plate timeline):
- edge: consecutive speed commands in one direction (gaps <= 0.25 s) with |dv| >= 0.2 m/s and rate >= 0.35 m/s^2
  (the start-up ramp is ~0.25 m/s^2); start-ups (from 0) and stops (to 0) are excluded.
- pulse: an edge followed within 6 s by an edge back to the starting speed; direction "slow" or "fast" from the first
  edge. An edge without a return is a single "step" event.
- transition window: 2 s before the onset to 8 s after the speed settles (end of the return edge), cut at the next
  event's onset. Steady window: from 10 s after the end of the start-up ramp to 2 s before the first event. Steady
  walking therefore comes from earlier in the trial than the transitions (time-in-trial is not controlled).
- during-pulse window: onset to 1 s after the speed settles. Errors rise mainly here and are back to baseline within
  ~1 s, so this is the headline transition metric; the long window above dilutes it.
- recovery: time after the speed settles until the 0.5 s centred RMS error first falls to <= 1.2 x that trial's steady
  RMS error (censored at the window end).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import preprocess as pp
from .config import project_path
from .evaluate import AXES, excursion_errors, moving_average, stride_frames

MIN_DV, MIN_RATE, MAX_GAP_S, PAIR_MAX_S = 0.2, 0.35, 0.25, 6.0
PRE_S, POST_S, STEADY_AFTER_START_S, RECOVERY_X, RMS_WIN_S = 2.0, 8.0, 10.0, 1.2, 0.5
PULSE_POST_S = 1.0           # 'during pulse' window: onset to 1 s after the speed settles
CURVE_PRE_S, CURVE_POST_S, CURVE_DT = 2.0, 12.0, 0.1


def speed_edges(tm: pd.DataFrame) -> pd.DataFrame:
    """All runs of same-direction speed commands, with kind 'start', 'stop' or 'change' (change = fast enough)."""
    t, v = tm["time"].to_numpy(float), tm["speed"].to_numpy(float) + 0.0
    groups = []
    for i in np.flatnonzero(np.diff(v) != 0) + 1:
        sgn = np.sign(v[i] - v[i - 1])
        if groups and groups[-1]["sgn"] == sgn and t[i] - groups[-1]["t_end"] <= MAX_GAP_S:
            groups[-1].update(t_end=t[i], v_to=v[i])
        else:
            groups.append({"sgn": sgn, "t_on": t[i], "t_end": t[i], "v_from": v[i - 1], "v_to": v[i]})
    rows = []
    for g in groups:
        dv = g["v_to"] - g["v_from"]
        if abs(dv) < MIN_DV:
            continue
        rate = abs(dv) / max(g["t_end"] - g["t_on"], 1 / pp.COM_RATE)
        kind = "start" if g["v_from"] <= 0.05 else "stop" if g["v_to"] <= 0.05 else "change" if rate >= MIN_RATE \
            else "slow_ramp"
        rows.append({"t_on": g["t_on"], "t_end": g["t_end"], "v_from": g["v_from"], "v_to": g["v_to"], "dv": dv,
                     "rate": rate, "kind": kind})
    return pd.DataFrame(rows, columns=["t_on", "t_end", "v_from", "v_to", "dv", "rate", "kind"])


def speed_events(edges: pd.DataFrame) -> pd.DataFrame:
    """Pair change edges into pulses; returns onset, settle (s), direction, dv, kind ('pulse' / 'step')."""
    ch = edges[edges.kind == "change"].reset_index(drop=True)
    out, i = [], 0
    while i < len(ch):
        e = ch.iloc[i]
        nxt = ch.iloc[i + 1] if i + 1 < len(ch) else None
        if nxt is not None and abs(nxt.v_to - e.v_from) < 1e-6 and nxt.t_on - e.t_end <= PAIR_MAX_S:
            out.append({"onset": e.t_on, "settle": nxt.t_end, "direction": "slow" if e.dv < 0 else "fast",
                        "dv": e.dv, "kind": "pulse"})
            i += 2
        else:
            out.append({"onset": e.t_on, "settle": e.t_end, "direction": "slow" if e.dv < 0 else "fast",
                        "dv": e.dv, "kind": "step"})
            i += 1
    ev = pd.DataFrame(out, columns=["onset", "settle", "direction", "dv", "kind"])
    if len(ev):
        nxt_onset = np.r_[ev.onset.to_numpy()[1:], np.inf]
        ev["end"] = np.minimum(ev.settle + POST_S, nxt_onset)
        ev["start"] = ev.onset - PRE_S
        half = len(ev) // 2
        ev["phase"] = ["early" if k < half else "late" for k in range(len(ev))]
    return ev


def trial_events(cfg: dict, subject: str, trial: str) -> tuple[pd.DataFrame, tuple[float, float] | None]:
    tm = pd.read_csv(project_path(cfg, "data_dir") / f"{subject}_{trial}_Treadmill.csv", engine="pyarrow")
    edges = speed_edges(tm)
    ev = speed_events(edges)
    starts = edges[edges.kind == "start"]
    t0 = (starts.t_end.iloc[0] if len(starts) else tm["time"].iloc[0]) + STEADY_AFTER_START_S
    t1 = (ev.onset.iloc[0] - PRE_S) if len(ev) else tm["time"].iloc[-1]
    return ev, ((t0, t1) if t1 > t0 else None)


def _frames(t: float, n: int) -> int:
    return int(np.clip(np.round(t * pp.COM_RATE), 0, n))


def score_trial(pred: np.ndarray, true: np.ndarray, mask: np.ndarray, events: pd.DataFrame, steady, stride: int):
    """Per-trial frame masks, squared errors and per-event records."""
    n = len(true)
    err = 1000 * (pred - np.nan_to_num(true))
    err[~mask] = np.nan
    exc = 1000 * excursion_errors(pred, np.where(mask[:, None], true, np.nan), mask, stride)
    in_tr = np.zeros(n, bool)
    in_pu = np.zeros(n, bool)
    in_st = np.zeros(n, bool)
    if steady is not None:
        in_st[_frames(steady[0], n):_frames(steady[1], n)] = True
    in_st &= mask
    steady_rms = np.sqrt(np.nanmean(err[in_st] ** 2, axis=0)) if in_st.any() else np.full(3, np.nan)
    roll = np.sqrt(moving_average(np.where(np.isfinite(err), err ** 2, np.nan), int(RMS_WIN_S * pp.COM_RATE)))
    lags = np.arange(int(-CURVE_PRE_S * pp.COM_RATE), int(CURVE_POST_S * pp.COM_RATE) + 1)
    records, curves = [], []
    for e in events.itertuples():
        a, b = _frames(e.start, n), _frames(e.end, n)
        if b <= a:
            continue
        in_tr[a:b] = True
        w = slice(a, b)
        o, s = _frames(e.onset, n), _frames(e.settle, n)
        in_pu[o:min(b, _frames(e.settle + PULSE_POST_S, n))] = True
        rec = {"onset_s": e.onset, "direction": e.direction, "kind": e.kind, "phase": e.phase, "dv": e.dv}
        for k, ax in enumerate(AXES):
            seg = roll[w, k]
            rec[f"peak_{ax}_mm"] = float(np.nanmax(seg)) if np.isfinite(seg).any() else np.nan
            after = roll[s:b, k]
            hit = np.flatnonzero(after <= RECOVERY_X * steady_rms[k]) if np.isfinite(steady_rms[k]) else []
            rec[f"recovery_{ax}_s"] = float(hit[0] / pp.COM_RATE) if len(hit) else np.nan
            rec[f"recovered_{ax}"] = bool(len(hit))
        records.append(rec)
        c = np.full((len(lags), 3), np.nan)
        idx = o + lags
        ok = (idx >= a) & (idx < b)
        c[ok] = np.abs(err[idx[ok]])
        curves.append((e.direction, c))
    in_tr &= mask
    in_pu &= mask
    return {"in_pu": in_pu, "in_tr": in_tr, "in_st": in_st, "err": err, "exc": exc, "records": records, "curves": curves,
            "lags_s": lags / pp.COM_RATE}


def score_run(cfg: dict, preds: dict, events_by_trial: dict, strides: dict) -> dict:
    """Pool a fold's test trials. preds: {trial name: {pred, true, mask}}."""
    sums = {k: [] for k in ("tr", "pu", "st", "tr_exc", "pu_exc", "st_exc")}
    records, curves = [], {"slow": [], "fast": []}
    lags = None
    for name, d in preds.items():
        ev, steady = events_by_trial[name]
        r = score_trial(d["pred"].astype(float), d["true"].astype(float), d["mask"], ev, steady, strides[name])
        sums["tr"].append(r["err"][r["in_tr"]])
        sums["pu"].append(r["err"][r["in_pu"]])
        sums["pu_exc"].append(r["exc"][r["in_pu"]])
        sums["st"].append(r["err"][r["in_st"]])
        sums["tr_exc"].append(r["exc"][r["in_tr"]])
        sums["st_exc"].append(r["exc"][r["in_st"]])
        for rec in r["records"]:
            rec["trial"] = name
        records += r["records"]
        for direction, c in r["curves"]:
            curves[direction].append(c)
        lags = r["lags_s"]
    out = {"n_events": len(records)}
    rmse = {}
    for key in sums:
        e = np.concatenate(sums[key]) if sums[key] else np.zeros((0, 3))
        rmse[key] = np.sqrt(np.nanmean(e ** 2, axis=0)) if len(e) else np.full(3, np.nan)
    for k, ax in enumerate(AXES):
        out[f"steady_rmse_{ax}_mm"] = float(rmse["st"][k])
        out[f"trans_rmse_{ax}_mm"] = float(rmse["tr"][k])
        out[f"trans_ratio_{ax}"] = float(rmse["tr"][k] / rmse["st"][k])
        out[f"pulse_rmse_{ax}_mm"] = float(rmse["pu"][k])
        out[f"pulse_ratio_{ax}"] = float(rmse["pu"][k] / rmse["st"][k])
        out[f"pulse_exc_{ax}_mm"] = float(rmse["pu_exc"][k])
        out[f"steady_exc_{ax}_mm"] = float(rmse["st_exc"][k])
        out[f"trans_exc_{ax}_mm"] = float(rmse["tr_exc"][k])
    out["steady_rmse_mean_mm"] = float(np.mean(rmse["st"]))
    out["trans_rmse_mean_mm"] = float(np.mean(rmse["tr"]))
    out["pulse_rmse_mean_mm"] = float(np.mean(rmse["pu"]))
    out["score_50_50_mm"] = 0.5 * (out["steady_rmse_mean_mm"] + out["pulse_rmse_mean_mm"])
    rec = pd.DataFrame(records)
    if len(rec):
        for ax in AXES:
            out[f"peak_{ax}_mm_median"] = float(rec[f"peak_{ax}_mm"].median())
            out[f"recovery_{ax}_s_median"] = float(rec[f"recovery_{ax}_s"].median())
            out[f"recovered_{ax}_frac"] = float(rec[f"recovered_{ax}"].mean())
        for col, vals in (("direction", ("slow", "fast")), ("phase", ("early", "late"))):
            for v in vals:
                sub = [r_ for r_ in records if r_[col] == v]
                out[f"n_{v}"] = len(sub)
    # breakdowns by direction and phase use the per-event windows
    out["curve_lags_s"] = lags[::int(CURVE_DT * pp.COM_RATE)].round(3).tolist() if lags is not None else []
    for direction, cs in curves.items():
        if cs:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m = np.nanmean(np.stack(cs), axis=0)[::int(CURVE_DT * pp.COM_RATE)]
            out[f"curve_{direction}"] = np.round(m, 2).tolist()
    out["events"] = rec.to_dict(orient="records")
    return out


def breakdown(preds: dict, events_by_trial: dict, strides: dict, by: str) -> dict:
    """During-pulse RMSE per axis split by an event column ('direction' or 'phase')."""
    out = {}
    for name, d in preds.items():
        ev, steady = events_by_trial[name]
        for value, sub in ev.groupby(by):
            r = score_trial(d["pred"].astype(float), d["true"].astype(float), d["mask"], sub, steady, strides[name])
            out.setdefault(value, []).append(r["err"][r["in_pu"]])
    return {v: np.sqrt(np.nanmean(np.concatenate(es) ** 2, axis=0)).tolist() for v, es in out.items()}


SUMMARY_COLS = ["steady_rmse_ap_mm", "pulse_rmse_ap_mm", "pulse_ratio_ap", "trans_rmse_ap_mm", "trans_ratio_ap",
                "steady_rmse_mean_mm", "pulse_rmse_mean_mm", "trans_rmse_mean_mm", "score_50_50_mm",
                "pulse_exc_ap_mm", "steady_exc_ap_mm", "peak_ap_mm_median",
                "recovery_ap_s_median", "recovered_ap_frac", "trans_rmse_ap_mm_slow", "trans_rmse_ap_mm_fast",
                "trans_rmse_ap_mm_early", "trans_rmse_ap_mm_late"]


def collect(runs_dir, skip_tags=("dev", "smoke")) -> pd.DataFrame:
    """One row per run with a transitions.json."""
    import json
    rows = []
    for p in runs_dir.glob("*/*/*/transitions.json"):
        model, tag, fold = p.parts[-4:-1]
        if any(s in tag for s in skip_tags):
            continue
        d = json.loads(p.read_text())
        rows.append({"model": model, "tag": tag, "fold": fold, "n_events": d["n_events"],
                     **{c: d.get(c, np.nan) for c in SUMMARY_COLS}})
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "tag"])
    out = pd.concat([g[SUMMARY_COLS].mean(), g[SUMMARY_COLS].std().add_suffix("_sd")], axis=1)
    out["folds"] = g.size()
    out["n_events"] = g.n_events.sum()
    return out.sort_values("score_50_50_mm").reset_index()


def events_for_trials(cfg: dict, trials) -> tuple[dict, dict]:
    events = {t.name: trial_events(cfg, t.subject, t.trial) for t in trials}
    strides = {t.name: stride_frames(t.events) for t in trials}
    return events, strides
