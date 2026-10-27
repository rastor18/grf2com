"""Synthetic dataset in the raw CSV format (4 development subjects + 1 locked), used by the data and smoke tests.
Each trial's forces and moments are also written as a C3D in grf_raw/ (volts, like the Qualisys export)."""
from __future__ import annotations

import struct

import numpy as np
import pandas as pd
import pytest

from grf2com.config import load_config

FS, COM_FS, DUR_S, STRIDE_S, STANCE = 1200, 60, 40.0, 1.1, 0.62
SUBJECTS = {"S01": 70.0, "S02": 80.0, "S03": 65.0, "S04": 90.0, "S05": 75.0}   # mass (kg); S05 is locked
TRIALS = ["T1", "T2"]
OFFSETS = {"Fz": 25.0, "Fx": 60.0, "Fy": 300.0, "Mx": 30.0, "My": 500.0, "Mz": 150.0}
PLATE_X = {"left": -460.0, "right": 460.0}
OFF_TREADMILL_S = 3.0      # last seconds: subject off the plates (tests the valid mask)
NAN_FRAMES = slice(600, 610)
SPEED_LOG = [(0.05, 0.0), (5.0, 1.0), (20.0, 1.25), (DUR_S - 0.5, 0.0)]


def _side(t: np.ndarray, shift: float, mass: float, rng) -> dict:
    phase = ((t - shift) % STRIDE_S) / STRIDE_S
    stance = phase < STANCE
    p = np.where(stance, phase / STANCE, 0.0)
    fz = np.where(stance, mass * 9.81 * (0.55 + 0.25 * np.sin(np.pi * p) ** 2) * np.sin(np.pi * p) ** 0.3, 0.0)
    cop_y = 1000 + 250 * p - 80
    return {"Fz": fz, "Fy": np.where(stance, 0.15 * fz * np.cos(np.pi * p), 0.0),
            "Fx": np.where(stance, 0.05 * fz, 0.0), "copy": cop_y, "stance": stance,
            "noise": rng.normal(0, 0.2, len(t))}


CAL = np.array([500.0, 500.0, 1000.0, 800.0, 400.0, 400.0])   # N/V, N*m/V (as in the real files)
# plate 1 = right on channels 7-12 with Mx / My swapped, plate 2 = left on 1-6 (as in the real files)
PLATE_CHANNELS = {"right": [7, 8, 9, 11, 10, 12], "left": [1, 2, 3, 4, 5, 6]}


def _c3d_param(gid: int, name: str, dtype: int, dims: list[int], data: bytes) -> bytes:
    body = struct.pack("<bB", dtype, len(dims)) + bytes(dims) + data + b"\x00"
    return struct.pack("<bb", len(name), gid) + name.encode() + struct.pack("<h", len(body) + 2) + body


def _c3d_group(gid: int, name: str) -> bytes:
    return struct.pack("<bb", len(name), -gid) + name.encode() + struct.pack("<h", 3) + b"\x00"


def write_c3d(path, fm: dict[str, np.ndarray], fs: int = FS, frame_rate: int = COM_FS, nan_tail: int = 7) -> None:
    """Minimal float C3D with 12 analog channels from {Fx_left: N, ..., Mz_right: N*m}; `nan_tail` NaN samples are
    appended like the QTM padding. Inverse of grf2com.c3d.read_raw_grf."""
    from grf2com.c3d import SIGN
    spf = fs // frame_rate
    n = len(next(iter(fm.values())))
    volts = np.zeros((n, 14), np.float32)
    for side, chans in PLATE_CHANNELS.items():
        for q, c, g in zip(("Fx", "Fy", "Fz", "Mx", "My", "Mz"), chans, CAL):
            volts[:, c - 1] = fm[f"{q}_{side}"] / g / SIGN / -1.0   # ANALOG.SCALE is -1
    n_frames = -(-(n + nan_tail) // spf)
    volts = np.vstack([volts, np.full((n_frames * spf - n, 14), np.nan, np.float32)])
    f32 = lambda v: np.asarray(v, "<f4").tobytes()   # noqa: E731
    i16 = lambda v: np.asarray(v, "<i2").tobytes()   # noqa: E731
    chan = np.array([PLATE_CHANNELS["right"], PLATE_CHANNELS["left"]])   # (plate, 6) -> stored column-major (6, 2)
    cal = np.stack([np.diag(CAL)] * 2)                                    # (plate, 6, 6) -> (6, 6, 2)
    params = (_c3d_group(1, "POINT") + _c3d_param(1, "USED", 2, [], i16(0)) + _c3d_param(1, "SCALE", 4, [], f32(-1))
              + _c3d_param(1, "RATE", 4, [], f32(frame_rate))
              + _c3d_group(2, "ANALOG") + _c3d_param(2, "USED", 2, [], i16(14))
              + _c3d_param(2, "SCALE", 4, [14], f32([-1.0] * 14)) + _c3d_param(2, "OFFSET", 2, [14], i16([0] * 14))
              + _c3d_param(2, "GEN_SCALE", 4, [], f32(1.0)) + _c3d_param(2, "RATE", 4, [], f32(fs))
              + _c3d_param(2, "LABELS", -1, [4, 2], b"Ch01Ch02")
              + _c3d_group(3, "FORCE_PLATFORM") + _c3d_param(3, "USED", 2, [], i16(2))
              + _c3d_param(3, "TYPE", 2, [2], i16([4, 4])) + _c3d_param(3, "CHANNEL", 2, [6, 2], i16(chan))
              + _c3d_param(3, "CAL_MATRIX", 4, [6, 6, 2], f32(np.transpose(cal, (0, 2, 1))))
              + _c3d_param(3, "MATRIX_STORE", -1, [8], b"BYCOLUMN"))
    params += b"\x00\x00"
    n_blocks = -(-(len(params) + 4) // 512)
    psec = (bytes([1, 0x50, n_blocks, 84]) + params).ljust(n_blocks * 512, b"\x00")
    data_block = 2 + n_blocks
    header = struct.pack("<BBHHHHHfHHf", 2, 0x50, 0, 14 * spf, 1, n_frames, 0, -1.0, data_block, spf, frame_rate)
    path.write_bytes(header.ljust(512, b"\x00") + psec + volts.astype("<f4").tobytes())


def write_dataset(root, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    data = root / "AE_grf2com_csv"
    data.mkdir(parents=True)
    (root / "grf_raw").mkdir()
    t = np.arange(int(DUR_S * FS)) / FS
    tc = np.arange(int(DUR_S * COM_FS)) / COM_FS
    off = t > DUR_S - OFF_TREADMILL_S
    index, events, info = [], [], []
    for i, (subj, mass) in enumerate(SUBJECTS.items()):
        info.append({"subjects": subj, "ages": 20 + i, "sexes": i % 2, "heights": 165 + 3 * i, "shoes": 8,
                     "dephys": 9, "feet": 25 + 0.5 * i, "legs": 85 + i, "mass": mass})
        for trial in TRIALS:
            cols = {"time": t}
            for side, shift in (("left", 0.0), ("right", STRIDE_S / 2)):
                g = _side(t, shift, mass, rng)
                fz = np.where(off, 0.0, g["Fz"])
                fx, fy = np.where(off, 0.0, g["Fx"]), np.where(off, 0.0, g["Fy"])
                cop_x = PLATE_X[side]
                mx, my = g["copy"] * fz / 1000, -cop_x * fz / 1000
                cols.update({f"Fx_{side}": fx + OFFSETS["Fx"] + g["noise"], f"Fy_{side}": fy + OFFSETS["Fy"],
                             f"Fz_{side}": fz + OFFSETS["Fz"] + g["noise"],
                             f"COPx_{side}": cop_x + 3000.0, f"COPy_{side}": g["copy"] + 5000.0,
                             f"COPz_{side}": 0.0, f"Mx_{side}": mx + OFFSETS["Mx"], f"My_{side}": my + OFFSETS["My"],
                             f"Mz_{side}": OFFSETS["Mz"] + 0 * fz})
                hs = np.arange(shift, DUR_S - OFF_TREADMILL_S - STRIDE_S, STRIDE_S)
                for k_, h in enumerate(hs, 1):
                    events.append({"subject": subj, "trial": trial, "event": f"HS_{side}", "k": k_,
                                   "sample_index": int(round(h * FS)) + 1})
                    events.append({"subject": subj, "trial": trial, "event": f"TO_{side}", "k": k_,
                                   "sample_index": int(round((h + STANCE * STRIDE_S) * FS)) + 1})
            order = ["time"] + [f"{q}_{s}" for q in ("Fx", "Fy", "Fz", "COPx", "COPy", "COPz", "Mx", "My", "Mz")
                                for s in ("left", "right")]
            pd.DataFrame(cols)[order].to_csv(data / f"{subj}_{trial}_GRF.csv", index=False)
            write_c3d(root / "grf_raw" / f"{subj}_T{int(trial[1:]):02d}.c3d",
                      {k: v for k, v in cols.items() if k[:2] in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")})
            com = np.column_stack([0.8 + 0.05 * np.sin(2 * np.pi * tc / 15) + 0.01 * i,
                                   0.03 * np.sin(2 * np.pi * tc / STRIDE_S),
                                   0.85 + 0.002 * (165 + 3 * i - 170) + 0.01 * np.sin(4 * np.pi * tc / STRIDE_S)])
            com[NAN_FRAMES] = np.nan
            pd.DataFrame({"frame": np.arange(1, len(tc) + 1), "time": tc, "CoM_Pos_1": com[:, 0],
                          "CoM_Pos_2": com[:, 1], "CoM_Pos_3": com[:, 2]}).to_csv(
                data / f"{subj}_{trial}_CoM.csv", index=False, na_rep="NaN")
            pd.DataFrame([{"time": a, "speed": b, "frame": a * 60 + 1, "label": b} for a, b in SPEED_LOG]).to_csv(
                data / f"{subj}_{trial}_Treadmill.csv", index=False)
            index.append({"subject": subj, "trial": trial, "n_grf_samples": len(t), "n_com_frames": len(tc)})
    pd.DataFrame(index).to_csv(data / "trial_index.csv", index=False)
    pd.DataFrame(events).to_csv(data / "grf_events.csv", index=False)
    pd.DataFrame(info).to_csv(root / "infotable.csv", index=False)


def synthetic_config(root, model: str | None = None, overrides=()) -> dict:
    from grf2com.config import ROOT
    cfg = load_config(ROOT / "configs" / "models" / f"{model}.yaml" if model else None, [
        f"paths.data_dir={(root / 'AE_grf2com_csv').as_posix()}", f"paths.raw_grf_dir={(root / 'grf_raw').as_posix()}",
        f"paths.infotable={(root / 'infotable.csv').as_posix()}",
        f"paths.trial_map={(root / 'trial_map.csv').as_posix()}", f"paths.cache_dir={(root / 'cache').as_posix()}",
        f"paths.runs_dir={(root / 'runs').as_posix()}", f"paths.results_dir={(root / 'results').as_posix()}",
        "data.final_test_subjects=[S05]", "data.final_test_notes={}", "preprocess.workers=1",
        "train.device=cpu", "train.batch_size=4", "train.epochs=2", "train.steps_per_epoch=3", "train.patience=5",
        "train.dev_epochs=2", "train.dev_steps_per_epoch=3", "eval.latency_runs=2", *overrides])
    return cfg


@pytest.fixture(scope="session")
def synth_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    write_dataset(root)
    from grf2com import preprocess as pp
    pp.run(synthetic_config(root), verbose=False)
    return root
