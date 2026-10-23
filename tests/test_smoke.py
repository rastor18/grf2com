"""End-to-end on the synthetic dataset: a tiny training loop, the baselines, the leaderboard guard and the final test."""
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from conftest import synthetic_config
from grf2com import preprocess as pp
from grf2com.baselines import BASELINES
from grf2com.config import ROOT, project_path
from grf2com.data import load_trials, make_folds
from grf2com.evaluate import score
from grf2com.final import run_final
from grf2com.tables import load_infotable
from grf2com.train import run_dir, train_fold


@pytest.mark.parametrize("model,vertical", [("tcn", "absolute"), ("gru", "absolute"), ("tcn", "anthro_centered")])
def test_tiny_training_loop(synth_root, model, vertical):
    cfg = synthetic_config(synth_root, model, [f"target.vertical={vertical}"])
    fold = make_folds(cfg, only=["S01"])[0]
    out = run_dir(cfg, f"smoke_{vertical}", fold.name)
    s = train_fold(cfg, fold, out, torch.device("cpu"), log=lambda *a: None)
    m = json.loads((out / "metrics.json").read_text())
    assert np.isfinite(m["test"]["rmse_mean_mm"]) and m["test_subjects"] == ["S01"]
    assert (out / "best.pt").exists() and (out / "trace.png").exists()
    assert s["best_epoch"] >= 1


def test_baselines_run(synth_root):
    cfg = synthetic_config(synth_root)
    info = load_infotable(cfg)
    fold = make_folds(cfg, only=["S02"])[0]
    train, val, test = (load_trials(cfg, getattr(fold, p)) for p in ("train", "val", "test"))
    for name, cls in BASELINES.items():
        model = cls(cfg).fit(train, val, info)
        res = score(test, [model.predict(t, info) for t in test])
        assert np.isfinite(res["pooled"]["rmse_mean_mm"]), name


def test_final_test_refuses_without_confirm():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "final_test.py")], capture_output=True, text=True)
    assert r.returncode == 2 and "Refusing" in r.stderr


def test_final_test_path(synth_root):
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "scripts"))
    from run_baselines import run_one

    cfg = synthetic_config(synth_root, "tcn")
    base = synthetic_config(synth_root)
    info = load_infotable(base)
    for fold in make_folds(cfg):
        train_fold(cfg, fold, run_dir(cfg, "full_s0", fold.name), torch.device("cpu"), log=lambda *a: None)
        for name in BASELINES:
            run_one(base, name, fold, info, pp.cache_dir(base), project_path(base, "runs_dir") / name / "full" / fold.name)
    out = synth_root / "results" / "final"
    for _ in range(2):
        table = run_final([cfg], base, "full_s0", "full", torch.device("cpu"), out, log=lambda *a: None)
    assert set(table.model) == {"tcn", *BASELINES}
    assert (out / "final.md").exists()
    md = (out / "final.md").read_text(encoding="utf-8")
    assert "look number 2" in md and "S05" in md


def test_single_belt_models_and_baselines(synth_root):
    over = ["data.features=[total_force,single_cop,single_moments,speed]"]
    cfg = synthetic_config(synth_root, "tcn", over)
    fold = make_folds(cfg, only=["S03"])[0]
    s = train_fold(cfg, fold, run_dir(cfg, "smoke_single", fold.name), torch.device("cpu"), log=lambda *a: None)
    assert np.isfinite(s["test"]["rmse_mean_mm"])
    base = synthetic_config(synth_root, overrides=over)
    info = load_infotable(base)
    train, val, test = (load_trials(base, getattr(fold, p)) for p in ("train", "val", "test"))
    for name, cls in BASELINES.items():
        if getattr(cls, "split_belt_only", False):
            continue
        model = cls(base).fit(train, val, info)
        assert np.isfinite(score(test, [model.predict(t, info) for t in test])["pooled"]["rmse_mean_mm"]), name


def test_estimator_state_features(synth_root):
    from grf2com.data import FeatureBuilder
    from grf2com.train import load_checkpoint
    cfg = synthetic_config(synth_root, "tcn", ["data.features=[forces,cop,moments,contact,speed,lab_kf,kalman_state]"])
    fold = make_folds(cfg, only=["S04"])[0]
    out = run_dir(cfg, "smoke_estimators", fold.name)
    s = train_fold(cfg, fold, out, torch.device("cpu"), log=lambda *a: None)
    assert np.isfinite(s["test"]["rmse_mean_mm"])
    _, fb, _, _ = load_checkpoint(out / "best.pt")            # the fitted Kalman travels with the checkpoint
    assert fb._kalman is not None and "kal_ap" in fb.names and "labkf_since_step" in fb.names
    test = load_trials(cfg, fold.test)
    assert np.isfinite(fb.transform(test[0])).all()
    assert isinstance(fb, FeatureBuilder)
