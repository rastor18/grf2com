"""Open-loop replay of the lab's self-paced treadmill controller with different state estimators.

The lab controller (self-paced-treadmill-VT_visual_feedback/src/treadmill_controller.py, speed_control_with_kalman)
acts once per detected step:
    v_target = v_tm_mean + gain_p (com_step - p_off) + gain_v dcom_step      (Song et al. 2020, eq. 4)
    a_target = |v_target - v_tm_mean| / delta_self_paced                      (eq. 5)
bounded to [min_v, max_v] and [a_min, max_a]. com_step / dcom_step are the subject's mean fore-aft position (lab frame,
treadmill centre = 0) and speed over the step just finished.

Open loop: recorded trials are replayed, so the subject never reacts to these commands and the belt in the data keeps
its fixed protocol. What is scored is the command each estimator *would* issue at each step, against an oracle
(the same control law fed with mocap CoM). Estimators:
- kf: the lab's Kalman filter (baselines/song_kf.run_filter, verified against kalman_filter.py), com_step - L/2.
- oracle: mocap CoM_Pos_1: step mean over (t_prev, t], speed = displacement / step time.
- net:<model>/<tag>: a trained network's streamed, out-of-fold AP prediction, same step statistics as the oracle.
Mocap and network positions are put in the controller's frame with one constant: the median over all replayed steps of
(KF com_step - mocap step mean). The controller sees step times and belt speed from the KF in every case, so only the
state estimate differs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import preprocess as pp
from .baselines import song_kf as sk

LAB_CONTROLLER = {"p_off": -0.05, "gain_p": 0.35, "gain_v": 0.42, "min_v": 0.0, "max_v": 2.6,
                  "delta_self_paced": 0.3, "max_a": 1.5, "a_min": 0.001}   # reset_speed_controller_param
TREADMILL_LENGTH = 1.8          # treadmill_dimension[0]; lab frame AP = Bertec COP AP - L / 2
WARMUP_S = 10.0                 # steps this soon after the filter starts are not scored (start-up transient)


def control_law(com_step, dcom_step, v_tm_mean, p: dict = LAB_CONTROLLER):
    """Target speed and acceleration, as in speed_control_with_kalman (gain schedule and far override off)."""
    dv = p["gain_p"] * (np.asarray(com_step) - p["p_off"]) + p["gain_v"] * np.asarray(dcom_step)
    v = np.clip(np.asarray(v_tm_mean) + dv, p["min_v"], p["max_v"])
    a = np.clip(np.abs(dv / p["delta_self_paced"]), p["a_min"], p["max_a"])
    return v, a


def _step_stats(y: np.ndarray, valid: np.ndarray, t_prev: np.ndarray, t_now: np.ndarray):
    """Step-mean position over frames in (t_prev, t_now] and speed = displacement / time (latest frame <= t)."""
    n = len(y)
    f0 = np.clip(np.floor(t_prev * pp.COM_RATE).astype(int), 0, n - 1)
    f1 = np.clip(np.floor(t_now * pp.COM_RATE).astype(int), 0, n - 1)
    yv = np.where(valid, y, np.nan)
    c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(yv))])
    k = np.concatenate([[0], np.cumsum(np.isfinite(yv))])
    cnt = k[f1 + 1] - k[f0 + 1]
    full = cnt == (f1 - f0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(full & (cnt > 0), (c[f1 + 1] - c[f0 + 1]) / cnt, np.nan)
        speed = (yv[f1] - yv[f0]) / (t_now - t_prev)
    return mean, speed


def replay_trial(cfg: dict, trial, net_preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per KF-detected step: KF / oracle / network com_step and dcom_step (frames not yet aligned)."""
    sig = sk.load_signals(cfg, trial.subject, trial.trial, pp.cache_dir(cfg))
    res = sk.run_filter(sig, sk.calibrated_mass(sig, trial.mass))
    st = res["steps"]
    if not len(st):
        return pd.DataFrame()
    t_now = res["t"][st.tick.to_numpy()]
    t_prev = np.r_[res["t"][0] - sk.LOOP_DT, t_now[:-1]]
    out = pd.DataFrame({"subject": trial.subject, "trial": trial.trial, "t": t_now, "step_t": st.step_t.to_numpy(),
                        "since_start": st.t.to_numpy(), "v_tm_mean": st.v_tm_mean.to_numpy(),
                        "kf_com": st.com_step.to_numpy() - TREADMILL_LENGTH / 2, "kf_dcom": st.dcom_step.to_numpy()})
    out["oracle_com"], out["oracle_dcom"] = _step_stats(trial.y[:, 0].astype(float), trial.mask, t_prev, t_now)
    for name, pred in net_preds.items():
        out[f"{name}_com"], out[f"{name}_dcom"] = _step_stats(pred[:, 0].astype(float), trial.mask, t_prev, t_now)
    return out


def score(steps: pd.DataFrame, estimators: list[str], p: dict = LAB_CONTROLLER) -> tuple[pd.DataFrame, float]:
    """Align frames, apply the control law per estimator and score each against the oracle."""
    use = steps[steps.since_start >= WARMUP_S].copy()
    offset = float(np.nanmedian(use.kf_com - use.oracle_com))
    for e in ["oracle"] + [e for e in estimators if e != "kf"]:
        use[f"{e}_com"] = use[f"{e}_com"] + offset
    for e in ["oracle"] + estimators:
        use[f"{e}_v"], use[f"{e}_a"] = control_law(use[f"{e}_com"], use[f"{e}_dcom"], use.v_tm_mean, p)
    return use, offset


def summarize(use: pd.DataFrame, estimators: list[str], by: str | None = None) -> pd.DataFrame:
    rows = []
    groups = use.groupby(by) if by else [("all", use)]
    for key, g in groups:
        ok = g[[f"{e}_v" for e in ["oracle"] + estimators]].notna().all(axis=1)
        g = g[ok]
        rough_o = np.sqrt(np.nanmean(np.diff(g.oracle_v) ** 2))
        for e in estimators:
            d = g[f"{e}_v"] - g.oracle_v
            dv_e, dv_o = g[f"{e}_v"] - g.v_tm_mean, g.oracle_v - g.v_tm_mean
            rows.append({by or "set": key, "estimator": e, "steps": len(g),
                         "cmd_rmse_mps": float(np.sqrt(np.mean(d ** 2))), "cmd_bias_mps": float(d.mean()),
                         "cmd_corr": float(np.corrcoef(dv_e, dv_o)[0, 1]),
                         "pos_term_rmse_mps": float(np.sqrt(np.mean((LAB_CONTROLLER["gain_p"] * (
                             g[f"{e}_com"] - g.oracle_com)) ** 2))),
                         "vel_term_rmse_mps": float(np.sqrt(np.mean((LAB_CONTROLLER["gain_v"] * (
                             g[f"{e}_dcom"] - g.oracle_dcom)) ** 2))),
                         "com_rmse_mm": float(1000 * np.sqrt(np.mean((g[f"{e}_com"] - g.oracle_com) ** 2))),
                         "dcom_rmse_mps": float(np.sqrt(np.mean((g[f"{e}_dcom"] - g.oracle_dcom) ** 2))),
                         "direction_agree": float(np.mean(np.sign(dv_e) == np.sign(dv_o))),
                         "roughness_mps": float(np.sqrt(np.mean(np.diff(g[f"{e}_v"]) ** 2))),
                         "oracle_roughness_mps": float(rough_o)})
    return pd.DataFrame(rows)
