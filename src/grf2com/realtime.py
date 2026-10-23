"""Sample-by-sample (real-time) versions of the preprocessing and of model inference.

StreamingPreprocessor reproduces preprocess.process_trial one raw sample at a time: tare, contact flag, COP recomputed
from moments and held while unloaded, causal Butterworth low-pass, and one 60 Hz frame (preprocess.CHANNELS) every
1 / grf_rate s. Optionally, a causal emulation of an upstream filter the training data had (realtime.upstream_filter,
e.g. the MATLAB filtfilt at 10 Hz) is applied to the raw forces and moments first.

StreamingEstimator keeps the last window of frames and runs the model on it at every frame, returning the CoM for the
newest frame (the full window of past context, as in real-time use).

tests/test_realtime.py checks that both match the batch pipeline exactly.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
import torch
from scipy import signal

from . import preprocess as pp
from .data import CH, FeatureBuilder, Trial

RAW = [f"{q}_{s}" for s in pp.SIDES for q in pp.TARED]          # Fx_l, Fy_l, ..., Mz_r (forces N, moments N*m)


FRAME = [c for c in pp.CHANNELS if c != "speed"]                 # filtered channels, in frame order
_IDX = {c: i for i, c in enumerate(RAW)}


class _BlockSOS:
    """Causal SOS filter over blocks of samples for several channels; each channel starts in steady state at its own
    first finite sample (NaN before it), like preprocess.causal_filter."""

    def __init__(self, sos: np.ndarray, n_ch: int):
        self.sos = sos
        self.zi = np.zeros((sos.shape[0], 2, n_ch))
        self.started = np.zeros(n_ch, bool)
        self.zi_unit = signal.sosfilt_zi(sos)[:, :, None]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """x: (n, n_ch) without NaN once a channel has started."""
        y = np.full_like(x, np.nan, dtype=float)
        if self.started.all():
            y, self.zi = signal.sosfilt(self.sos, x, axis=0, zi=self.zi)
            return y
        for c in range(x.shape[1]):
            col = x[:, c]
            if not self.started[c]:
                finite = np.flatnonzero(np.isfinite(col))
                if not len(finite):
                    continue
                i0 = finite[0]
                self.zi[:, :, c] = self.zi_unit[:, :, 0] * col[i0]
                self.started[c] = True
            else:
                i0 = 0
            y[i0:, c], self.zi[:, :, c] = signal.sosfilt(self.sos, col[i0:], zi=self.zi[:, :, c])
        return y


def _ffill(x: np.ndarray, carry: np.ndarray) -> np.ndarray:
    """Forward-fill NaN down the rows of x, starting from `carry` (one value per column; NaN = nothing yet)."""
    out = np.vstack([carry[None], x])
    idx = np.where(np.isfinite(out), np.arange(len(out))[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return out[idx, np.arange(out.shape[1])][1:]


class StreamingPreprocessor:
    """Feed raw samples with push(); returns a frame (len(preprocess.CHANNELS),) when one is due, else None.

    Samples are buffered and processed as one block when a frame is due (on the sample at the frame time), so the
    output is identical to processing them one by one, at a fraction of the cost."""

    def __init__(self, pcfg: dict, offsets: dict[str, float], fs: float = pp.GRF_RATE,
                 upstream_filter: dict | None = None):
        rate = int(pcfg["grf_rate"])
        self.frame_dt = 1.0 / rate
        self.eps = 0.25 / fs    # tolerate rounded sample times (a quarter sample)
        self.off = np.array([float(offsets.get(c, 0.0)) for c in RAW])
        self.threshold = float(pcfg["contact_threshold_n"])
        if not bool(pcfg.get("cop_from_moments", True)):
            raise ValueError("streaming preprocessing recomputes COP from moments (preprocess.cop_from_moments: true)")
        sos = signal.butter(int(pcfg["filter_order"]), float(pcfg["lowpass_hz"]), fs=fs, output="sos")
        self.lowpass = _BlockSOS(sos, len(FRAME))
        self.pre = []
        if upstream_filter:
            pre = signal.butter(int(upstream_filter["order"]), float(upstream_filter["cutoff_hz"]), fs=fs, output="sos")
            self.pre = [_BlockSOS(pre, len(RAW)) for _ in range(int(upstream_filter.get("passes_live", 1)))]
        self.raw_carry = np.zeros(len(RAW))
        self.cop_carry = np.full(4, np.nan)          # COPx_l, COPx_r, COPy_l, COPy_r
        self.t0 = None
        self.next_frame = 0
        self.buf_raw, self.buf_speed = [], []

    def push(self, t: float, raw: dict[str, float], speed: float) -> np.ndarray | None:
        if self.t0 is None:
            self.t0 = t
        self.buf_raw.append([raw[c] for c in RAW])
        self.buf_speed.append(speed)
        if t - self.t0 + self.eps < self.next_frame * self.frame_dt:
            return None
        self.next_frame += 1
        frame = self._process(np.asarray(self.buf_raw, float))[-1]
        speed_now = self.buf_speed[-1] + 0.0
        self.buf_raw, self.buf_speed = [], []
        return np.append(frame, speed_now).astype(np.float32)

    def _process(self, x: np.ndarray) -> np.ndarray:
        x = _ffill(x, self.raw_carry)
        self.raw_carry = x[-1]
        for f in self.pre:
            x = f(x)
        tared = x - self.off
        out = np.empty((len(x), len(FRAME)))
        for s in pp.SIDES:
            fz0 = tared[:, _IDX[f"Fz_{s}"]]
            contact = fz0 > self.threshold
            for q in pp.TARED:
                out[:, CH[f"{q}_{s}"]] = tared[:, _IDX[f"{q}_{s}"]]
            out[:, CH[f"contact_{s}"]] = contact
            with np.errstate(divide="ignore", invalid="ignore"):
                out[:, CH[f"COPx_{s}"]] = np.where(contact, -1000 * tared[:, _IDX[f"My_{s}"]] / fz0, np.nan)
                out[:, CH[f"COPy_{s}"]] = np.where(contact, 1000 * tared[:, _IDX[f"Mx_{s}"]] / fz0, np.nan)
        cop = [CH["COPx_l"], CH["COPx_r"], CH["COPy_l"], CH["COPy_r"]]
        out[:, cop] = _ffill(out[:, cop], self.cop_carry)
        self.cop_carry = out[-1, cop]
        return self.lowpass(out)


class StreamingEstimator:
    """Runs a trained model on the rolling window of frames; push(frame) returns the CoM (3,) in m for that frame."""

    LIVE = "LIVE"

    def __init__(self, model, fb: FeatureBuilder, cfg: dict, subject: dict):
        unsupported = {"lab_kf", "kalman_state"} & set(fb.groups)
        if unsupported:
            raise ValueError(f"feature groups {sorted(unsupported)} are not available in the streaming runtime yet")
        self.model = model.cpu().eval()
        self.fb = fb
        self.k = int(cfg["preprocess"]["grf_rate"]) // pp.COM_RATE
        self.window = int(round(cfg["eval"]["window_s"] * pp.COM_RATE)) * self.k
        self.info = pd.DataFrame([subject], index=[self.LIVE])
        fb._info = self.info
        self.mass = float(subject["mass"])
        self.s = torch.from_numpy(fb.subject_vector(self.LIVE, self.info))[None]
        self.buf: deque = deque(maxlen=self.window)

    @torch.inference_mode()
    def push(self, frame: np.ndarray) -> np.ndarray:
        self.buf.append(frame)
        x = np.stack(self.buf)
        n_frames = len(x) // self.k if len(x) >= self.k else 1
        t = Trial(self.LIVE, "live", x, np.full((n_frames, 3), np.nan, np.float32), np.zeros(n_frames, bool), {},
                  self.mass)
        feats = torch.from_numpy(self.fb.transform(t))[None]
        z = self.model(feats, self.s)[:, ::self.k][0, -1].numpy()[None]
        return self.fb.denormalize(z, self.LIVE)[0]


def stream_trial(pre: StreamingPreprocessor, t: np.ndarray, raw: dict[str, np.ndarray], speed: np.ndarray,
                 estimator: StreamingEstimator | None = None):
    """Run a whole recorded trial through the streaming pipeline; returns (frames, predictions or None)."""
    frames, preds = [], []
    for i in range(len(t)):
        f = pre.push(t[i], {c: raw[c][i] for c in RAW}, speed[i])
        if f is not None:
            frames.append(f)
            if estimator is not None:
                preds.append(estimator.push(f))
    return np.array(frames), (np.array(preds) if estimator is not None else None)


__all__ = ["RAW", "StreamingEstimator", "StreamingPreprocessor", "stream_trial", "CH"]
