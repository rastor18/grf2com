"""Deployable model bundles: one directory with the trained weights, the fitted feature state and everything the
real-time runtime needs. Bundles are trained on the development subjects only (never the locked final-test subjects).

<bundle>/model.pt     checkpoint (train.save_checkpoint: weights, FeatureBuilder state, full config)
<bundle>/bundle.json  metadata: model, training subjects, inputs, rates, upstream filter, required subject fields, ...
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .anthro import HEIGHT_FEATURES
from .config import ROOT, project_path
from .train import _jsonable, load_checkpoint

BUNDLE_VERSION = 1


def _git_revision() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             timeout=5)
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True,
                               timeout=5)
        return (out.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")) if out.returncode == 0 else None
    except Exception:
        return None


def required_subject_fields(fb) -> list[str]:
    fields = list(fb.subject_features) + ["mass"]
    if fb.vertical != "absolute":
        fields += HEIGHT_FEATURES
    return sorted(set(fields))


def write_bundle(out_dir: Path, checkpoint: Path, cfg: dict, summary: dict, loso: dict | None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, out_dir / "model.pt")
    _, fb, _, _ = load_checkpoint(out_dir / "model.pt")
    meta = {
        "bundle_version": BUNDLE_VERSION, "created": dt.datetime.now().isoformat(timespec="seconds"),
        "code_revision": _git_revision(), "model": cfg["model"]["name"], "config_file": cfg.get("config_file"),
        "train_subjects": summary["train_subjects"], "n_train_trials": summary["n_train_trials"],
        "epochs": summary["epochs_run"], "n_params": summary["n_params"],
        "cpu_latency_ms": summary.get("cpu_latency_ms"), "inputs": fb.names, "feature_groups": fb.groups,
        "subject_features": fb.subject_features, "vertical": fb.vertical,
        "required_subject_fields": required_subject_fields(fb),
        "grf_rate_hz": cfg["preprocess"]["grf_rate"], "window_s": cfg["eval"]["window_s"],
        "preprocess": cfg["preprocess"], "upstream_filter": (cfg.get("realtime") or {}).get("upstream_filter"),
        "output": "CoM_Pos_1..3 (m), mocap lab frame; one value per 60 Hz frame",
        "development_loso": loso,
    }
    (out_dir / "bundle.json").write_text(json.dumps(_jsonable(meta), indent=1))
    return out_dir


def load_bundle(path: str | Path):
    """Returns (model on CPU, FeatureBuilder, config, metadata)."""
    path = Path(path)
    model, fb, cfg, _ = load_checkpoint(path / "model.pt", "cpu")
    meta = json.loads((path / "bundle.json").read_text())
    return model.eval(), fb, cfg, meta


def loso_summary(cfg: dict, model: str, tag: str) -> tuple[dict | None, int | None]:
    """Mean development LOSO test metrics and the median best epoch of a tag's runs."""
    runs = [json.loads(p.read_text()) for p in (project_path(cfg, "runs_dir") / model / tag).glob("*/metrics.json")]
    if not runs:
        return None, None
    keys = ["rmse_mean_mm", "rmse_ap_mm", "rmse_ml_mm", "rmse_vert_mm", "exc_rmse_mean_mm"]
    summ = {"tag": tag, "folds": len(runs), **{k: float(np.mean([r["test"][k] for r in runs])) for k in keys}}
    return summ, int(round(float(np.median([r["best_epoch"] for r in runs]))))
