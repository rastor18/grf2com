"""Synthetic dataset in the raw CSV format (4 development subjects + 1 locked), used by the data and smoke tests."""
from __future__ import annotations

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


def write_dataset(root, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    data = root / "AE_grf2com_csv"
    data.mkdir(parents=True)
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
        f"paths.data_dir={(root / 'AE_grf2com_csv').as_posix()}", f"paths.infotable={(root / 'infotable.csv').as_posix()}",
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
