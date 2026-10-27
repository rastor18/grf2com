"""Minimal C3D reader for the raw force-plate recordings in grf_raw/ (Qualisys exports, analog data only).

read_c3d returns the header, the parameters ({group: {name: value}}) and the analog channels in volts after
(stored - OFFSET) * SCALE * GEN_SCALE. read_raw_grf turns a trial into the CSV GRF columns (Fx..Mz per side, N and
N*m), unfiltered: the CSV export is a zero-phase 10 Hz Butterworth (filtfilt) of exactly these signals
(scripts/check_raw_grf.py checks it per trial).
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocess import GRF_RATE, SIDES, TARED

INTEL = 84
# FORCE_PLATFORM order in these files: plate 1 is the right belt, plate 2 the left (matched against the CSV export).
PLATE_OF_SIDE = {"right": 0, "left": 1}
# The CSV reports the force on the subject; the plate calibration gives the force on the plate.
SIGN = -1.0


def _parameters(buf: bytes, start: int) -> dict:
    """Parameter section starting at byte `start` -> {GROUP: {NAME: value}} (names upper case)."""
    if buf[start + 3] != INTEL:
        raise ValueError(f"only Intel-format C3D files are supported (processor type {buf[start + 3]})")
    end = start + buf[start + 2] * 512
    groups, params = {}, []
    i = start + 4
    while i < end:
        nlen, gid = struct.unpack_from("<bb", buf, i)
        if nlen == 0:
            break
        name = buf[i + 2:i + 2 + abs(nlen)].decode("latin1").upper()
        j = i + 2 + abs(nlen)
        nxt = struct.unpack_from("<h", buf, j)[0]
        k = j + 2
        if gid < 0:
            groups[-gid] = name
        else:
            dtype, ndim = struct.unpack_from("<bB", buf, k)
            dims = list(buf[k + 2:k + 2 + ndim])
            k += 2 + ndim
            n = int(np.prod(dims)) if dims else 1
            raw = buf[k:k + abs(dtype) * n]
            if dtype == -1:
                if len(dims) >= 2 and dims[0]:
                    w = dims[0]
                    value = [raw[x:x + w].decode("latin1").strip() for x in range(0, len(raw), w)]
                else:
                    value = raw.decode("latin1").strip()
            else:
                fmt = {1: np.uint8, 2: np.int16, 4: np.float32}[dtype]
                arr = np.frombuffer(raw, fmt).astype(float)
                value = arr.reshape(dims[::-1]).transpose() if len(dims) > 1 else (arr if dims else arr[0])
            params.append((gid, name, value))
        if nxt == 0:
            break
        i = j + nxt
    out: dict = {g: {} for g in groups.values()}
    for gid, name, value in params:
        out.setdefault(groups.get(gid, str(gid)), {})[name] = value
    return out


def read_c3d(path: str | Path) -> tuple[dict, dict, np.ndarray]:
    """Returns (header, parameters, analog (n_samples, n_channels) in volts)."""
    buf = Path(path).read_bytes()
    (param_block, _key, n_points, analog_per_frame, first, last, _gap, scale, data_block, samples_per_frame,
     rate) = struct.unpack_from("<BBHHHHHfHHf", buf, 0)
    params = _parameters(buf, (param_block - 1) * 512)
    n_frames = last - first + 1
    long_frames = params.get("POINT", {}).get("LONG_FRAMES")
    if long_frames is not None and int(long_frames) > n_frames:
        n_frames = int(long_frames)
    if scale >= 0:
        raise ValueError(f"{path}: integer C3D data is not supported (only float, POINT.SCALE < 0)")
    n_ch = analog_per_frame // samples_per_frame if samples_per_frame else 0
    per_frame = 4 * n_points + analog_per_frame
    data = np.frombuffer(buf, "<f4", count=n_frames * per_frame, offset=(data_block - 1) * 512)
    analog = data.reshape(n_frames, per_frame)[:, 4 * n_points:].reshape(n_frames * samples_per_frame, n_ch)
    a = params["ANALOG"]
    offset = np.broadcast_to(np.asarray(a.get("OFFSET", 0.0), float), (n_ch,))
    ch_scale = np.broadcast_to(np.asarray(a.get("SCALE", 1.0), float), (n_ch,))
    volts = (analog.astype(float) - offset) * ch_scale * float(a.get("GEN_SCALE", 1.0))
    header = {"n_points": n_points, "n_analog_channels": n_ch, "first_frame": first, "last_frame": last,
              "n_frames": n_frames, "point_rate": rate, "analog_rate": float(a.get("RATE", rate * samples_per_frame)),
              "samples_per_frame": samples_per_frame}
    return header, params, volts


def c3d_name(subject: str, trial: str) -> str:
    """AE02, T1 -> AE02_T01.c3d"""
    return f"{subject}_T{int(trial.lstrip('T')):02d}.c3d"


def read_raw_grf(path: str | Path, n_samples: int | None = None) -> pd.DataFrame:
    """Raw (unfiltered) forces and moments with the CSV GRF column names: time, Fx_left, ..., Mz_right.

    Trailing all-NaN rows (QTM pads the last frame) are dropped; `n_samples` (the CSV row count) truncates further."""
    header, params, volts = read_c3d(path)
    if header["analog_rate"] != GRF_RATE:
        raise ValueError(f"{path}: analog rate {header['analog_rate']} Hz, expected {GRF_RATE}")
    fp = params["FORCE_PLATFORM"]
    if not np.all(np.asarray(fp["TYPE"]) == 4):
        raise ValueError(f"{path}: force plate type {fp['TYPE']} is not supported (type 4, 6x6 calibration)")
    if str(fp.get("MATRIX_STORE", "BYCOLUMN")).upper() != "BYCOLUMN":
        raise ValueError(f"{path}: FORCE_PLATFORM.MATRIX_STORE {fp['MATRIX_STORE']} is not supported")
    chan = np.asarray(fp["CHANNEL"], int)              # (6, n_plates), 1-based
    cal = np.asarray(fp["CAL_MATRIX"], float)          # (6, 6, n_plates): [F; M] = cal @ V
    finite = np.isfinite(volts).all(axis=1)
    n = int(np.flatnonzero(finite)[-1]) + 1 if finite.any() else 0
    if n_samples is not None:
        if n_samples > n:
            raise ValueError(f"{path}: {n} finite samples, fewer than the {n_samples} requested")
        n = n_samples
    cols = {"time": np.arange(n) / GRF_RATE}
    for side, p in PLATE_OF_SIDE.items():
        fm = SIGN * volts[:n, chan[:, p] - 1] @ cal[:, :, p].T
        for q, x in zip(TARED, fm.T):
            cols[f"{q}_{side}"] = x
    return pd.DataFrame(cols)


__all__ = ["c3d_name", "read_c3d", "read_raw_grf", "PLATE_OF_SIDE", "SIDES"]
