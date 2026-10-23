"""Live (shadow-mode) operation: sample sources, tare, and the run loop that logs and publishes CoM estimates.

Sources yield (t [s], raw {RAW channel: value}, belt speed [m/s]):
- ReplaySource: a recorded development trial from the raw CSVs, optionally paced in real time (locked subjects are
  refused).
- BertecSource: the Bertec ZMQ force-data publisher (the same stream bertec_remote_control.py reads), subscribed
  WITHOUT conflation so every message is used. Read-only: it never sends treadmill commands and does not need the
  RPC connection, so it can run next to the lab software.
"""
from __future__ import annotations

import csv
import json
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import preprocess as pp
from .config import project_path
from .realtime import RAW, StreamingEstimator, StreamingPreprocessor
from .tables import final_test_subjects


class ReplaySource:
    def __init__(self, cfg: dict, subject: str, trial: str, pace: float = 0.0):
        if subject in final_test_subjects(cfg):
            raise ValueError(f"{subject} is a locked final-test subject")
        stem = project_path(cfg, "data_dir") / f"{subject}_{trial}"
        grf = pd.read_csv(f"{stem}_GRF.csv", engine="pyarrow")
        tm = pd.read_csv(f"{stem}_Treadmill.csv", engine="pyarrow")
        self.t = grf["time"].to_numpy(float)
        self.raw = {c: grf[f"{c.split('_')[0]}_{pp.SIDES[c[-1]]}"].to_numpy(float) for c in RAW}
        idx = np.searchsorted(tm["time"].to_numpy(float), self.t, side="right") - 1
        self.speed = np.where(idx >= 0, tm["speed"].to_numpy(float)[np.maximum(idx, 0)], 0.0)
        self.pace = pace
        self.fs = pp.GRF_RATE

    def __iter__(self):
        start = time.perf_counter()
        for i in range(len(self.t)):
            if self.pace:
                wait = (self.t[i] - self.t[0]) / self.pace - (time.perf_counter() - start)
                if wait > 0.001:
                    time.sleep(wait)
            yield self.t[i], {c: self.raw[c][i] for c in RAW}, self.speed[i]


class SyntheticSource:
    """Made-up walking forces for testing without hardware (live_gui.py's "Synthetic" source): plates unloaded but
    with un-zeroed offsets like the real ones for `unloaded_s`, then alternating stance at a fixed stride time and belt
    speed. Not a gait model: it only exercises the pipeline, so the CoM estimates it produces mean nothing.
    pace 1 = real time with t = time.perf_counter() (like BertecSource), 0 = as fast as possible."""

    OFFSETS = {"Fx_l": 68.0, "Fx_r": 146.0, "Fy_l": 295.0, "Fy_r": 291.0, "Fz_l": 21.0, "Fz_r": 21.0}

    def __init__(self, mass: float = 76.5, fs: float = 1000.0, stride_s: float = 1.1, speed: float = 1.25,
                 unloaded_s: float = 4.0, pace: float = 1.0):
        self.bw, self.fs, self.stride, self.speed = 9.81 * mass, fs, stride_s, speed
        self.unloaded_s, self.pace = unloaded_s, pace
        self.first_keys = None

    def sample(self, t: float) -> dict[str, float]:
        raw = {c: self.OFFSETS.get(c, 0.0) for c in RAW}
        if t < self.unloaded_s:
            return raw
        ph = ((t - self.unloaded_s) % self.stride) / self.stride
        for side, shift, x0 in (("l", 0.0, -465.0), ("r", 0.5, 465.0)):
            p = (ph - shift) % 1.0
            if p >= 0.6:   # swing
                continue
            s = p / 0.6
            fz = 1.1 * self.bw * np.sin(np.pi * s)
            cop_y = 750.0 + 300.0 * s                                   # mm from the rear edge, heel -> toe
            raw[f"Fz_{side}"] += fz
            raw[f"Fy_{side}"] += 0.15 * fz * np.cos(np.pi * s)          # braking, then propulsion (reaction = -Fy)
            raw[f"Fx_{side}"] += (0.05 if side == "l" else -0.05) * fz  # reaction toward the midline
            raw[f"Mx_{side}"] += fz * cop_y / 1000.0                    # COPy = 1000 Mx / Fz
            raw[f"My_{side}"] += -fz * x0 / 1000.0                      # COPx = -1000 My / Fz
        return raw

    def __iter__(self):
        start = time.perf_counter()
        i = 0
        while True:
            rel = i / self.fs
            if self.pace:
                wait = start + rel / self.pace - time.perf_counter()
                if wait > 0.002:
                    time.sleep(wait)
            yield start + rel, self.sample(rel), (self.speed if rel >= self.unloaded_s else 0.0)
            i += 1

    def close(self):
        pass


class BertecSource:
    def __init__(self, bcfg: dict, host: str | None = None, port: int | None = None, timeout_ms: int = 5000):
        import zmq   # pyzmq; the lab PC has it for bertec_remote_control.py

        self.channels = {c: (k, float(sign)) for c, (k, sign) in bcfg["channels"].items()}
        missing = sorted(set(RAW) - set(self.channels))
        if missing:
            raise ValueError(f"realtime.bertec.channels is missing {missing}")
        self.speed_keys = list(bcfg["speed_keys"])
        self.address = f"tcp://{host or bcfg['host']}:{port or bcfg['data_port']}"
        self.timeout_ms = timeout_ms
        self._again = zmq.Again
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.connect(self.address)
        self.sock.subscribe("")
        self.first_keys: list[str] | None = None
        self.fs: float | None = None

    def __iter__(self):
        while True:
            try:
                msg = self.sock.recv_json()
            except self._again:
                raise TimeoutError(f"no Bertec data on {self.address} for {self.timeout_ms / 1000:.0f} s") from None
            if self.first_keys is None:
                self.first_keys = sorted(msg)
            t = time.perf_counter()
            raw = {c: sign * float(msg[k]) for c, (k, sign) in self.channels.items()}
            yield t, raw, float(np.mean([msg[k] for k in self.speed_keys]))

    def close(self):
        self.sock.close(0)


def measure_offsets(samples, seconds: float) -> tuple[dict[str, float], float]:
    """Median of each raw channel over `seconds` (plates unloaded) and the measured sample rate."""
    buf, t0, last = [], None, None
    for t, raw, _ in samples:
        t0 = t if t0 is None else t0
        buf.append([raw[c] for c in RAW])
        last = t
        if t - t0 >= seconds:
            break
    arr = np.array(buf)
    fs = (len(arr) - 1) / (last - t0) if last and last > t0 else float("nan")
    return dict(zip(RAW, np.median(arr, axis=0).tolist())), fs


def run(samples, pre: StreamingPreprocessor, est: StreamingEstimator, log_path: Path, udp: str | None = None,
        duration: float | None = None, log_raw: bool = False, truth=None, on_frame=None) -> dict:
    """Shadow-mode loop: preprocess every sample, estimate on every frame, log (and optionally publish over UDP)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sock, addr = None, None
    if udp:
        host, port = udp.rsplit(":", 1)
        sock, addr = socket.socket(socket.AF_INET, socket.SOCK_DGRAM), (host, int(port))
    cols = ["t", "frame", "com_ap", "com_ml", "com_vert", "compute_ms"] + (RAW + ["speed"] if log_raw else [])
    stats = {"samples": 0, "frames": 0, "compute_ms": []}
    t_first = None
    with open(log_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for t, raw, speed in samples:
            t_first = t if t_first is None else t_first
            stats["samples"] += 1
            frame = pre.push(t, raw, speed)
            if frame is not None:
                c0 = time.perf_counter()
                com = est.push(frame)
                ms = 1000 * (time.perf_counter() - c0)
                stats["compute_ms"].append(ms)
                row = [f"{t:.4f}", stats["frames"], *(f"{v:.5f}" for v in com), f"{ms:.2f}"]
                if log_raw:
                    row += [f"{raw[c]:.4f}" for c in RAW] + [f"{speed:.3f}"]
                w.writerow(row)
                if sock:
                    sock.sendto(json.dumps({"t": t, "frame": stats["frames"], "com": com.tolist()}).encode(), addr)
                if on_frame:
                    on_frame(stats["frames"], com)
                stats["frames"] += 1
            if duration and t - t_first >= duration:
                break
    ms = np.array(stats["compute_ms"]) if stats["compute_ms"] else np.zeros(1)
    elapsed = (t - t_first) if t_first is not None else 0.0
    return {"samples": stats["samples"], "frames": stats["frames"], "elapsed_s": elapsed,
            "input_rate_hz": stats["samples"] / elapsed if elapsed else float("nan"),
            "compute_ms_median": float(np.median(ms)), "compute_ms_p99": float(np.percentile(ms, 99)),
            "compute_ms_max": float(ms.max())}
