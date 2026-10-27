"""Song et al. (2020) treadmill state estimator, replaying the lab's implementation as a baseline for AP CoM position.

Reference implementation: self-paced-treadmill-VT_visual_feedback/src/kalman_filter.py (KalmanFilter.initKalman and
kalman_filter_step), run by treadmill_controller.speed_control_with_kalman on a 15 ms timer. Method: S Song, H Choi,
SH Collins, J NeuroEng Rehabil 17:68 (2020).

Every loop iteration (15 ms, Variables.DT_SPEED_CONTROLL) the state x = [position, speed] (fore-aft, COP frame) is
predicted with acceleration u = -(Fy_l + Fy_r) / m, with A, B, Q built once for delt = 0.01 s as in initKalman (only
the belt drift of the stored foot positions uses the measured loop time). A new step is a plate whose Fz / m rises
above 0.3 g. At a step: com_mes = midpoint of the new foot's COP and the previous foot's COP carried back by the belt,
dcom_mes = step length / step time - mean belt speed; the innovation is against the filter's running means over the
step (com_step, dcom_step). Steps of 3.6 s or longer are not used, except the first after initialization.
R = diag(0.025, 0.085)^2, P0 and sigma_a^2 = 2.9 as in the code. The filter starts when the treadmill starts.

Replay choices for this offline data (all causal):
- Force-plate samples are the tared 1200 Hz channels (the lab uses Bertec AUTOZERO data), taken as the latest sample at
  each loop tick; no extra filtering (the lab applies none, and the exported data are already filtered).
- COP is the COP recomputed from the tared moments (this pipeline's default), in m; the lab uses Bertec's COP.
- Mass is measured from the plates while standing before the treadmill starts (the lab's mass calibration), falling
  back to the infotable mass.
- Belt speed is the commanded speed log (the lab reads the measured belt speed).
- The estimate is in COP coordinates; one offset to lab-frame CoM_Pos_1 is fit on the training subjects (the lab
  frame shift plus the COP-to-CoM offset). The method estimates fore-aft only: ML and vertical come from the mean
  baseline (training mean, height regression).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import preprocess as pp
from ..anthro import SubjectHeight
from ..config import config_hash, project_path

LOOP_DT = 0.015              # Variables.DT_SPEED_CONTROLL: actual period of kalman_filter_step
MODEL_DT = 0.01              # delt used for A, B, Q in initKalman
NOISE_Q2 = 2.9
NOISE_V = (0.025, 0.085)     # R = noise_v noise_v'
P0 = ((0.0035, 0.0015), (0.0015, 0.0016))
THRESHOLD_FZ = 0.3 * 9.81    # on Fz / m (m s^-2), i.e. 30 % of body weight
MAX_STEP_S = 3.6             # gate (the code notes it was widened from 1.2 s)
FY_SIGN = -1.0               # subject acceleration = -Fy / m (code and data agree)
VERSION = 5


def load_signals(cfg: dict, subject: str, trial: str, cdir: Path) -> dict:
    """1200 Hz inputs: tared Fy, Fz (N) and fore-aft COP (m) per plate, commanded belt speed, time."""
    stem = project_path(cfg, "data_dir") / f"{subject}_{trial}"
    tare = json.loads((cdir / f"{subject}_{trial}.json").read_text())
    # the same force source (CSV or raw C3D, prefiltered or not) that the tare offsets were measured on
    grf = pp.load_grf(subject, trial, cfg["preprocess"], project_path(cfg, "data_dir"),
                      project_path(cfg, "raw_grf_dir"))
    tm = pd.read_csv(f"{stem}_Treadmill.csv", engine="pyarrow")
    out = {"t": grf["time"].to_numpy(float)}
    for s, side in pp.SIDES.items():
        for q in ("Fy", "Fz", "Mx"):
            out[f"{q}_{s}"] = grf[f"{q}_{side}"].to_numpy(float) - tare[f"off_{q}_{s}"]
        if cfg["preprocess"].get("cop_from_moments", True):
            with np.errstate(divide="ignore", invalid="ignore"):
                out[f"COPy_{s}"] = out[f"Mx_{s}"] / out[f"Fz_{s}"]          # N*m / N = m
        else:
            out[f"COPy_{s}"] = grf[f"COPy_{side}"].to_numpy(float) / 1000
    idx = np.searchsorted(tm["time"].to_numpy(float), out["t"], side="right") - 1
    out["speed"] = np.where(idx >= 0, tm["speed"].to_numpy(float)[np.maximum(idx, 0)], 0.0)
    return out


def start_index(sig: dict) -> int | None:
    """First sample with a nonzero belt-speed command (the lab initializes the filter when the treadmill starts)."""
    moving = np.flatnonzero(sig["speed"] > 0)
    return int(moving[0]) if len(moving) else None


def calibrated_mass(sig: dict, fallback_kg: float) -> float:
    """Mass from total Fz while standing before the treadmill starts (the lab's mass calibration)."""
    i0 = start_index(sig)
    total = sig["Fz_l"] + sig["Fz_r"]
    standing = np.zeros(len(total), bool)
    if i0:
        standing[:i0] = (total[:i0] > 0.5 * fallback_kg * pp.G) & (sig["Fz_l"][:i0] > 0) & (sig["Fz_r"][:i0] > 0)
    return float(np.median(total[standing]) / pp.G) if standing.sum() >= pp.GRF_RATE else float(fallback_kg)


def run_filter(sig: dict, mass_kg: float, fs: float = pp.GRF_RATE, loop_dt: float = LOOP_DT) -> dict:
    """Replay kalman_filter_step on a fixed loop_dt tick from the treadmill start. Returns tick times, the state after
    each tick (p, v) and the step log."""
    i0 = start_index(sig)
    empty = {"t": np.zeros(0), "p": np.zeros(0), "v": np.zeros(0), "steps": pd.DataFrame()}
    if i0 is None:
        return empty
    t_start = sig["t"][i0]
    ticks = np.arange(t_start + loop_dt, sig["t"][-1], loop_dt)   # first tick one timer period after init
    rows = np.minimum(np.searchsorted(sig["t"], ticks, side="right") - 1, len(sig["t"]) - 1)
    fy = FY_SIGN * (sig["Fy_l"] + sig["Fy_r"])[rows] / mass_kg
    fzl, fzr = sig["Fz_l"][rows] / mass_kg, sig["Fz_r"][rows] / mass_kg
    copl, copr, vtms = sig["COPy_l"][rows], sig["COPy_r"][rows], sig["speed"][rows]

    d = MODEL_DT
    a01, b0, b1 = d, d * d / 2, d                               # A = [[1, d], [0, 1]], B = [d^2/2, d]
    q00, q01, q11 = b0 * b0 * NOISE_Q2, b0 * b1 * NOISE_Q2, b1 * b1 * NOISE_Q2
    r0, r1 = NOISE_V[0] ** 2, NOISE_V[1] ** 2
    x0 = x1 = 0.0
    (p00, p01), (_, p11) = P0
    p_step_0 = p_step_1 = 0.0
    t_step_0, t_step_1 = -0.5, 0.0
    len_step, n_step, n_ts = 0.7, 0, 0
    com_step = dcom_step = v_tm_mean = 0.0
    stance_l = stance_r = False
    global_t1 = 0.0
    out_p, out_v = np.empty(len(ticks)), np.empty(len(ticks))
    steps = []
    for k in range(len(ticks)):
        now = ticks[k] - t_start
        vtm = vtms[k]
        global_t0, global_t1 = global_t1, now
        delt = global_t1 - global_t0
        # time update: x = A x + B u, P = A P A' + Q
        u = fy[k]
        x0, x1 = x0 + a01 * x1 + b0 * u, x1 + b1 * u
        p00, p01, p11 = (p00 + 2 * a01 * p01 + a01 * a01 * p11 + q00, p01 + a01 * p11 + q01, p11 + q11)
        com_step = (com_step * n_ts + x0) / (n_ts + 1)
        dcom_step = (dcom_step * n_ts + x1) / (n_ts + 1)
        v_tm_mean = (v_tm_mean * n_ts + vtm) / (n_ts + 1)
        n_ts += 1
        flag = False
        if not stance_l and fzl[k] > THRESHOLD_FZ:
            flag, p_step_0, p_step_1, leg = True, p_step_1, copl[k], "l"
        if not stance_r and fzr[k] > THRESHOLD_FZ:
            flag, p_step_0, p_step_1, leg = True, p_step_1, copr[k], "r"
        if flag:
            t_step_0, t_step_1 = t_step_1, now
            len_step = p_step_1 - p_step_0
            n_step += 1
            n_ts = 0
        stance_l, stance_r = fzl[k] > THRESHOLD_FZ, fzr[k] > THRESHOLD_FZ
        p_step_0 -= vtm * delt
        p_step_1 -= vtm * delt
        if flag:
            step_t = t_step_1 - t_step_0
            used = n_step == 1 or step_t < MAX_STEP_S
            com_mes = (p_step_1 + p_step_0) / 2
            dcom_mes = len_step / step_t - v_tm_mean if step_t > 0 else np.nan
            if used and np.isfinite(com_mes) and np.isfinite(dcom_mes):
                # K = P (P + R)^-1;  x += K (y_mes - y_est);  P = (I - K) P
                s00, s01, s11 = p00 + r0, p01, p11 + r1
                det = s00 * s11 - s01 * s01
                i00, i01, i11 = s11 / det, -s01 / det, s00 / det
                k00, k01 = p00 * i00 + p01 * i01, p00 * i01 + p01 * i11
                k10, k11 = p01 * i00 + p11 * i01, p01 * i01 + p11 * i11
                e0, e1 = com_mes - com_step, dcom_mes - dcom_step
                x0, x1 = x0 + k00 * e0 + k01 * e1, x1 + k10 * e0 + k11 * e1
                p00, p01, p11 = ((1 - k00) * p00 - k01 * p01, (1 - k00) * p01 - k01 * p11,
                                 -k10 * p01 + (1 - k11) * p11)
            else:
                used = False
            # com_step / dcom_step / v_tm_mean are what speed_control_with_kalman reads on this tick (the running means
            # over the step just finished; the measurement update changes x but not these)
            steps.append({"t": now, "tick": k, "leg": leg, "step_t": step_t, "com_mes": com_mes, "dcom_mes": dcom_mes,
                          "used": used, "com_step": com_step, "dcom_step": dcom_step, "v_tm_mean": v_tm_mean})
        out_p[k], out_v[k] = x0, x1
    return {"t": ticks, "p": out_p, "v": out_v, "steps": pd.DataFrame(steps)}


class LabKalmanStream:
    """kalman_filter_step one tick at a time, for live use (the live GUI's lab-KF model): the arithmetic of run_filter
    (tests/test_song_kf.py checks they agree tick for tick). As in the lab code, a tick whose force input hasn't changed
    since the previous one is skipped (no new data; skip_repeated=False turns this off, as in run_filter)."""

    def __init__(self, mass_kg: float, skip_repeated: bool = True):
        self.mass, self.skip_repeated = float(mass_kg), skip_repeated
        d = MODEL_DT
        self.a01, self.b0, self.b1 = d, d * d / 2, d
        self.q00, self.q01, self.q11 = self.b0 * self.b0 * NOISE_Q2, self.b0 * self.b1 * NOISE_Q2, self.b1 * self.b1 * NOISE_Q2
        self.r0, self.r1 = NOISE_V[0] ** 2, NOISE_V[1] ** 2
        self.x0 = self.x1 = 0.0
        (self.p00, self.p01), (_, self.p11) = P0
        self.p_step_0 = self.p_step_1 = 0.0
        self.t_step_0, self.t_step_1 = -0.5, 0.0
        self.len_step, self.n_step, self.n_ts = 0.7, 0, 0
        self.com_step = self.dcom_step = self.v_tm_mean = 0.0
        self.stance_l = self.stance_r = False
        self.global_t1 = 0.0
        self.u_prev = None

    def tick(self, now: float, fy_sum_n: float, fz_l_n: float, fz_r_n: float, cop_l: float, cop_r: float,
             vtm: float) -> dict | None:
        """One loop iteration at `now` (s since the filter started) on the latest sample: tared Fy_l + Fy_r and Fz per
        plate (N), fore-aft COP per plate (m), belt speed. Returns the step record on a step tick (`used`: the
        controller acts on it), else None; skipped ticks change nothing."""
        u = FY_SIGN * fy_sum_n / self.mass
        if self.skip_repeated and self.u_prev is not None and u == self.u_prev:
            return None
        self.u_prev = u
        fzl, fzr = fz_l_n / self.mass, fz_r_n / self.mass
        global_t0, self.global_t1 = self.global_t1, now
        delt = self.global_t1 - global_t0
        a01, b0, b1 = self.a01, self.b0, self.b1
        self.x0, self.x1 = self.x0 + a01 * self.x1 + b0 * u, self.x1 + b1 * u
        self.p00, self.p01, self.p11 = (self.p00 + 2 * a01 * self.p01 + a01 * a01 * self.p11 + self.q00,
                                        self.p01 + a01 * self.p11 + self.q01, self.p11 + self.q11)
        n = self.n_ts
        self.com_step = (self.com_step * n + self.x0) / (n + 1)
        self.dcom_step = (self.dcom_step * n + self.x1) / (n + 1)
        self.v_tm_mean = (self.v_tm_mean * n + vtm) / (n + 1)
        self.n_ts += 1
        flag, leg = False, None
        if not self.stance_l and fzl > THRESHOLD_FZ:
            flag, self.p_step_0, self.p_step_1, leg = True, self.p_step_1, cop_l, "l"
        if not self.stance_r and fzr > THRESHOLD_FZ:
            flag, self.p_step_0, self.p_step_1, leg = True, self.p_step_1, cop_r, "r"
        if flag:
            self.t_step_0, self.t_step_1 = self.t_step_1, now
            self.len_step = self.p_step_1 - self.p_step_0
            self.n_step += 1
            self.n_ts = 0
        self.stance_l, self.stance_r = fzl > THRESHOLD_FZ, fzr > THRESHOLD_FZ
        self.p_step_0 -= vtm * delt
        self.p_step_1 -= vtm * delt
        if not flag:
            return None
        step_t = self.t_step_1 - self.t_step_0
        used = self.n_step == 1 or step_t < MAX_STEP_S
        com_mes = (self.p_step_1 + self.p_step_0) / 2
        dcom_mes = self.len_step / step_t - self.v_tm_mean if step_t > 0 else np.nan
        if used and np.isfinite(com_mes) and np.isfinite(dcom_mes):
            s00, s01, s11 = self.p00 + self.r0, self.p01, self.p11 + self.r1
            det = s00 * s11 - s01 * s01
            i00, i01, i11 = s11 / det, -s01 / det, s00 / det
            k00, k01 = self.p00 * i00 + self.p01 * i01, self.p00 * i01 + self.p01 * i11
            k10, k11 = self.p01 * i00 + self.p11 * i01, self.p01 * i01 + self.p11 * i11
            e0, e1 = com_mes - self.com_step, dcom_mes - self.dcom_step
            self.x0, self.x1 = self.x0 + k00 * e0 + k01 * e1, self.x1 + k10 * e0 + k11 * e1
            p00, p01, p11 = self.p00, self.p01, self.p11
            self.p00, self.p01, self.p11 = ((1 - k00) * p00 - k01 * p01, (1 - k00) * p01 - k01 * p11,
                                            -k10 * p01 + (1 - k11) * p11)
        else:
            used = False
        return {"t": now, "leg": leg, "step_t": step_t, "com_mes": com_mes, "dcom_mes": dcom_mes, "used": used,
                "com_step": self.com_step, "dcom_step": self.dcom_step, "v_tm_mean": self.v_tm_mean}


def estimator_cache(cfg: dict) -> Path:
    key = {"version": VERSION, "loop_dt": LOOP_DT, "model_dt": MODEL_DT, "q2": NOISE_Q2, "noise_v": NOISE_V, "P0": P0,
           "threshold_fz": THRESHOLD_FZ, "max_step_s": MAX_STEP_S, "fy_sign": FY_SIGN,
           "cop_from_moments": cfg["preprocess"].get("cop_from_moments", True)}
    return pp.cache_dir(cfg) / f"song_kf_{config_hash(key, 8)}"


def estimate(cfg: dict, trial) -> dict:
    """Per-frame estimator output for a trial (latest tick at or before each CoM frame), cached; fold-independent."""
    path = estimator_cache(cfg) / f"{trial.subject}_{trial.trial}.npz"
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    sig = load_signals(cfg, trial.subject, trial.trial, pp.cache_dir(cfg))
    mass = calibrated_mass(sig, trial.mass)
    res = run_filter(sig, mass)
    frame_t = sig["t"][np.minimum(np.arange(trial.n_frames) * pp.FRAME_STEP, len(sig["t"]) - 1)]
    k = np.searchsorted(res["t"], frame_t, side="right") - 1
    have = k >= 0
    out = {"p": np.full(trial.n_frames, np.nan), "v": np.full(trial.n_frames, np.nan), "mass": np.array(mass)}
    out["p"][have], out["v"][have] = res["p"][k[have]], res["v"][k[have]]
    st = res["steps"]
    out["since_step"] = np.full(trial.n_frames, np.nan)          # seconds since the last detected step (causal)
    if len(st):
        step_times = res["t"][st.tick.to_numpy()]
        j = np.searchsorted(step_times, frame_t, side="right") - 1
        ok = j >= 0
        out["since_step"][ok] = frame_t[ok] - step_times[j[ok]]
    out["n_steps"] = np.array(len(st))
    out["n_used"] = np.array(int(st.used.sum()) if len(st) else 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out


class SongBaseline:
    name = "song_kf"
    split_belt_only = True    # needs per-belt foot contacts

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def fit(self, train, val, info: pd.DataFrame) -> "SongBaseline":
        self.mean = np.concatenate([t.y[t.mask] for t in train]).mean(0)
        self.height = SubjectHeight().fit(train, info)
        resid = []
        for t in train:
            p = estimate(self.cfg, t)["p"]
            ok = t.mask & np.isfinite(p)
            resid.append(t.y[ok, 0] - p[ok])
        self.offset = float(np.mean(np.concatenate(resid)))   # COP frame -> lab-frame CoM_Pos_1
        self.n_params = 1 + 2 + 3
        return self

    def predict(self, trial, info: pd.DataFrame) -> np.ndarray:
        out = np.tile(self.mean, (trial.n_frames, 1))
        p = estimate(self.cfg, trial)["p"]
        out[:, 0] = np.where(np.isfinite(p), p + self.offset, self.mean[0])
        out[:, 2] = self.height.predict(trial.subject, info)
        return out
