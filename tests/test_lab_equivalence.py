"""song_kf.run_filter must match the lab's own kalman_filter.py tick by tick on real data.

Imports the lab code (read-only) with stub PySide6 / variables modules and a simulated clock. Skipped when the lab repo
(paths.lab_repo) or the raw data are not available.
"""
import importlib
import sys
import types

import numpy as np
import pytest

from grf2com import preprocess as pp
from grf2com.baselines import song_kf as sk
from grf2com.config import load_config, project_path
from grf2com.data import load_trials

SECONDS = 60.0


class _Signal:
    def __init__(self, *a):
        pass

    def emit(self, *a):
        pass

    def connect(self, *a):
        pass


def _lab_kalman_module(lab_src):
    qtcore = types.ModuleType("PySide6.QtCore")
    qtcore.Signal, qtcore.QObject = _Signal, type("QObject", (), {"__init__": lambda self, *a: None})
    pyside = types.ModuleType("PySide6")
    pyside.QtCore = qtcore
    variables = types.ModuleType("variables")
    variables.Variables = type("Variables", (), {
        "data_vec_treadmillForce": np.zeros(12), "data_xy_copLeft": np.zeros(2), "data_xy_copRight": np.zeros(2),
        "mass": 80.0, "file": types.SimpleNamespace(write=lambda *a: None)})
    saved = {k: sys.modules.get(k) for k in ("PySide6", "PySide6.QtCore", "variables", "kalman_filter")}
    sys.modules.update({"PySide6": pyside, "PySide6.QtCore": qtcore, "variables": variables})
    sys.modules.pop("kalman_filter", None)
    sys.path.insert(0, str(lab_src))
    try:
        mod = importlib.import_module("kalman_filter")
    finally:
        sys.path.remove(str(lab_src))
        for k, v in saved.items():
            if k != "kalman_filter":
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
    return mod, variables.Variables


def test_port_matches_lab_kalman_filter():
    cfg = load_config()
    lab_src = project_path(cfg, "lab_repo") / "src"
    data = project_path(cfg, "data_dir") / "AE02_T1_GRF.csv"
    if not (lab_src / "kalman_filter.py").exists() or not data.exists():
        pytest.skip("lab repo or raw data not available")
    tr = load_trials(cfg, [("AE02", "T1")])[0]
    sig = sk.load_signals(cfg, "AE02", "T1", pp.cache_dir(cfg))
    i0 = sk.start_index(sig)
    stop = np.searchsorted(sig["t"], sig["t"][i0] + SECONDS)
    sig = {k: v[:stop] for k, v in sig.items()}
    mass = sk.calibrated_mass(sig, tr.mass)
    mine = sk.run_filter(sig, mass)

    mod, V = _lab_kalman_module(lab_src)
    clock = {"t": sig["t"][i0]}
    mod.time = types.SimpleNamespace(time=lambda: clock["t"])
    V.mass = mass
    kf = mod.KalmanFilter()
    kf.initKalman()
    rows = np.minimum(np.searchsorted(sig["t"], mine["t"], side="right") - 1, len(sig["t"]) - 1)
    lab_p, lab_v, n_steps = [], [], 0
    for k, (tick, r) in enumerate(zip(mine["t"], rows)):
        clock["t"] = tick
        f = np.zeros(12)
        f[0], f[2], f[6], f[8] = sig["Fy_l"][r], -sig["Fz_l"][r], sig["Fy_r"][r], -sig["Fz_r"][r]
        V.data_vec_treadmillForce = f
        V.data_xy_copLeft = np.array([sig["COPy_l"][r], 0.0])
        V.data_xy_copRight = np.array([sig["COPy_r"][r], 0.0])
        kf.kalman_filter_step(sig["speed"][r])
        lab_p.append(kf.kalman_variables["x"][0, 0])
        lab_v.append(kf.kalman_variables["x"][1, 0])
        n_steps += kf.kalman_variables["flag_step"]
    np.testing.assert_allclose(mine["p"], lab_p, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(mine["v"], lab_v, rtol=1e-7, atol=1e-9)
    assert n_steps == len(mine["steps"]) > 50
