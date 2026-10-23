"""Causal steady-state Kalman filter per axis: double-integrated force as the motion model, corrected by a
position measurement.

State [p, v] at 60 Hz. Input: acceleration a = g * (F/m - EMA(F/m)), where F is the summed force on that axis
(AP <- Fy, ML <- Fx, vertical <- Fz), the causal EMA (2 s) removes gravity and offsets, and the gain g is fit by
least squares against the mocap CoM second difference on training subjects.
Measurement: AP / ML - a linear map of causally low-passed (2 Hz) per-plate COP (combined single-belt COP when the
config uses single-belt inputs), fit on training subjects;
vertical - the anthropometric height estimate from the mean baseline.
Process / measurement noise (q, r) are chosen per axis on the validation subject.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import linalg, signal

from .. import preprocess as pp
from ..data import CH, combined_cop, is_single_belt, rate_mult
from ..anthro import SubjectHeight

DT = 1.0 / pp.COM_RATE
EMA_S = 2.0
COP_LOWPASS_HZ = 2.0
FORCE = {0: "Fy", 1: "Fx", 2: "Fz"}
COP_COLS = ["COPx_l", "COPx_r", "COPy_l", "COPy_r"]
Q_GRID = np.logspace(-5, 4, 10)
R_GRID = np.logspace(-6, 1, 8)


def ema(x: np.ndarray, tau_s: float) -> np.ndarray:
    alpha = DT / tau_s
    return signal.lfilter([alpha], [1, alpha - 1], x, zi=[x[0] * (1 - alpha)])[0]


def steady_state_tf(q: float, r: float):
    """Transfer functions from (u[t-1], z[t]) to the filtered position p[t]."""
    A = np.array([[1, DT], [0, 1]])
    B = np.array([[DT ** 2 / 2], [DT]])
    H = np.array([[1.0, 0.0]])
    P = linalg.solve_discrete_are(A.T, H.T, q * (B @ B.T), np.array([[r]]))
    K = P @ H.T / (H @ P @ H.T + r)
    IKH = np.eye(2) - K @ H
    F, G = IKH @ A, np.hstack([IKH @ B, K])
    C, D = np.array([[1.0, 0.0]]), np.zeros((1, 2))
    return [signal.ss2tf(F, G, C, D, input=i) for i in range(2)]


def run_filter(tfs, u: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Filtered position, starting from p = z[0], v = 0 (exact by linearity: the z -> p DC gain is 1)."""
    u_prev = np.concatenate([[0.0], u[:-1]])
    (bu, au), (bz, az) = tfs
    return signal.lfilter(bu[0], au, u_prev) + signal.lfilter(bz[0], az, z - z[0]) + z[0]


class KalmanBaseline:
    name = "kalman"

    def __init__(self, cfg: dict, qr: dict | None = None):
        """qr: fixed {axis: (q, r)} instead of tuning (used by the final test, which has no validation subject)."""
        self.cfg = cfg
        self.k = rate_mult(cfg)
        self.sos = signal.butter(2, COP_LOWPASS_HZ, fs=pp.COM_RATE, output="sos")
        self.fixed_qr = {int(a): tuple(v) for a, v in qr.items()} if qr else None
        self.single = is_single_belt(cfg)
        self.belt_offset = float(cfg["data"].get("belt_ml_offset_mm", 508.0))

    def _accel_input(self, t, axis: int) -> np.ndarray:
        x = t.x[::self.k]
        f = (x[:, CH[f"{FORCE[axis]}_l"]] + x[:, CH[f"{FORCE[axis]}_r"]]).astype(np.float64) / t.mass
        return f - ema(f, EMA_S)

    def _cop(self, t) -> np.ndarray:
        x = t.x[::self.k]
        if self.single:
            cx, cy, _ = combined_cop(x, self.belt_offset)
            v = np.stack([cx, cy], 1)
            v = np.where(np.isfinite(v), v, np.nanmean(v, axis=0))
            return np.stack([signal.sosfilt(self.sos, c, zi=signal.sosfilt_zi(self.sos) * c[0])[0] for c in v.T], 1)
        cols = []
        for c in COP_COLS:
            v = x[:, CH[c]].astype(np.float64)
            v = np.where(np.isfinite(v), v, self.cop_center[c])
            cols.append(signal.sosfilt(self.sos, v, zi=signal.sosfilt_zi(self.sos) * v[0])[0])
        return np.stack(cols, 1)

    def _measurement(self, t, axis: int, info: pd.DataFrame) -> np.ndarray:
        if axis == 2:
            return np.full(t.n_frames, self.height.predict(t.subject, info))
        c = self._cop(t)
        return np.column_stack([np.ones(len(c)), c]) @ self.meas_w[axis]

    def fit(self, train, val, info: pd.DataFrame) -> "KalmanBaseline":
        self.cop_center = {c: float(np.nanmean(np.concatenate([t.x[::self.k, CH[c]] for t in train])))
                           for c in COP_COLS}
        self.height = SubjectHeight().fit(train, info)
        self.gain, self.meas_w = {}, {}
        for axis in range(3):
            num = den = 0.0
            for t in train:
                y, m = t.y[:, axis].astype(np.float64), t.mask
                acc = np.full(t.n_frames, np.nan)
                acc[1:-1] = (y[2:] - 2 * y[1:-1] + y[:-2]) / DT ** 2
                ok = m.copy()
                ok[1:-1] &= m[2:] & m[:-2]
                ok[[0, -1]] = False
                f = self._accel_input(t, axis)
                num += np.sum(f[ok] * acc[ok])
                den += np.sum(f[ok] ** 2)
            self.gain[axis] = num / den
            if axis < 2:
                A = np.concatenate([np.column_stack([np.ones(t.n_frames), self._cop(t)])[t.mask] for t in train])
                b = np.concatenate([t.y[t.mask, axis] for t in train])
                self.meas_w[axis] = np.linalg.lstsq(A, b, rcond=None)[0]
        tune = val if val else train
        self.qr = dict(self.fixed_qr) if self.fixed_qr else {}
        for axis in range(3):
            if axis in self.qr:
                continue
            inputs = [(self.gain[axis] * self._accel_input(t, axis), self._measurement(t, axis, info), t)
                      for t in tune]
            best = None
            for q in Q_GRID:
                for r in R_GRID:
                    tfs = steady_state_tf(q, r)
                    err = np.concatenate([(run_filter(tfs, u, z) - t.y[:, axis])[t.mask] for u, z, t in inputs])
                    rmse = float(np.sqrt(np.mean(err ** 2)))
                    if best is None or rmse < best[0]:
                        best = (rmse, q, r)
            self.qr[axis] = best[1:]
        self.tfs = {axis: steady_state_tf(*self.qr[axis]) for axis in range(3)}
        self.n_params = 3 + 2 * 5 + 6 + 3
        return self

    def predict(self, trial, info: pd.DataFrame) -> np.ndarray:
        return np.column_stack([run_filter(self.tfs[a], self.gain[a] * self._accel_input(trial, a),
                                           self._measurement(trial, a, info)) for a in range(3)])
