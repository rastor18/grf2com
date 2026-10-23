"""Final test on the locked subjects: retrain on every development trial, score the locked subjects once.

Each network trains for the median best epoch of its development LOSO runs (no validation subject is needed). The
baselines reuse the settings their development runs chose (median ridge alpha, per-axis median Kalman q and r).
Every call appends to <out_root>/log.csv, so repeated looks at the locked subjects stay visible.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import preprocess as pp
from .baselines import BASELINES
from .config import project_path
from .data import final_fold, load_trials, make_folds
from .evaluate import score
from .reporting import md_table
from .tables import development_subjects, final_test_subjects, load_infotable, load_trial_index
from .train import _jsonable, train_fold

RANGE_FEATURES = ["heights", "mass", "legs", "feet"]


def _dev_metrics(cfg: dict, model: str, tag: str) -> list[dict]:
    paths = sorted((project_path(cfg, "runs_dir") / model / tag).glob("*/metrics.json"))
    return [json.loads(p.read_text()) for p in paths]


def _check_complete(cfg: dict, runs: list[dict], what: str, allow_incomplete: bool) -> None:
    n = len(make_folds(cfg))
    if not runs or (len(runs) < n and not allow_incomplete):
        raise RuntimeError(f"{what}: {len(runs)}/{n} development folds finished; finish the LOSO runs first "
                           "(or pass allow_incomplete)")


def subject_notes(cfg: dict) -> dict[str, list[str]]:
    """Why each locked subject may score differently: anthropometrics outside the development range, config notes."""
    info = load_infotable(cfg)
    dev = development_subjects(cfg, load_trial_index(cfg).subject)
    lo, hi = info.loc[dev, RANGE_FEATURES].min(), info.loc[dev, RANGE_FEATURES].max()
    notes = {}
    for s in final_test_subjects(cfg):
        out = []
        for f in RANGE_FEATURES:
            v = info.loc[s, f]
            if v < lo[f] or v > hi[f]:
                out.append(f"{f} {v:g} outside the development range {lo[f]:g}–{hi[f]:g}")
        extra = (cfg["data"].get("final_test_notes") or {}).get(s)
        if extra:
            out.append(extra)
        notes[s] = out
    return notes


def run_final(model_cfgs: list[dict], baseline_cfg: dict | None, tag: str, baseline_tag: str, device,
              out_root: Path, allow_incomplete: bool = False, log=print) -> pd.DataFrame:
    out_root.mkdir(parents=True, exist_ok=True)
    rows, per_subject = [], []
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    for cfg in model_cfgs:
        name = cfg["model"]["name"]
        runs = _dev_metrics(cfg, name, tag)
        _check_complete(cfg, runs, f"{name}/{tag}", allow_incomplete)
        epochs = max(1, int(round(float(np.median([r["best_epoch"] for r in runs])))))
        cfg = json.loads(json.dumps(cfg))
        cfg["train"]["epochs"] = epochs
        fold = final_fold(cfg)
        log(f"[final] {name}/{tag}: training on {len(fold.train)} development trials for {epochs} epochs "
            f"(median best epoch of {len(runs)} folds), testing on {fold.subjects('test')}")
        s = train_fold(cfg, fold, out_root / f"{name}_{tag}", device, log=log)
        rows.append({"model": name, "tag": tag, "epochs": epochs, **s["test"]})
        per_subject += [{"model": name, "tag": tag, **r} for r in s["per_subject"]]
    if baseline_cfg is not None:
        cfg = baseline_cfg
        fold = final_fold(cfg)
        info = load_infotable(cfg)
        cdir = pp.cache_dir(cfg)
        train, test = load_trials(cfg, fold.train, cdir), load_trials(cfg, fold.test, cdir)
        for name, cls in BASELINES.items():
            runs = _dev_metrics(cfg, name, baseline_tag)
            _check_complete(cfg, runs, f"{name}/{baseline_tag}", allow_incomplete)
            kwargs = {}
            if name == "ridge":
                kwargs["alpha"] = float(np.median([r["alpha"] for r in runs]))
            if name == "kalman":
                kwargs["qr"] = {a: tuple(float(np.exp(np.median([np.log(r["qr"][a][i]) for r in runs])))
                                         for i in range(2)) for a in ("0", "1", "2")}
            model = cls(cfg, **kwargs).fit(train, [], info)
            res = score(test, [model.predict(t, info) for t in test], full=True)
            d = out_root / f"{name}_{baseline_tag}"
            d.mkdir(parents=True, exist_ok=True)
            res["per_trial"].to_csv(d / "per_trial.csv", index=False)
            (d / "metrics.json").write_text(json.dumps(_jsonable(
                {"test": res["pooled"], "per_subject": res["per_subject"].to_dict(orient="records"),
                 "settings": {k: (v if not isinstance(v, dict) else {str(a): b for a, b in v.items()})
                              for k, v in kwargs.items()}}), indent=1))
            rows.append({"model": name, "tag": baseline_tag, "epochs": None, **res["pooled"]})
            per_subject += [{"model": name, "tag": baseline_tag, **r}
                            for r in res["per_subject"].to_dict(orient="records")]
            log(f"[final] {name}: RMSE mean {res['pooled']['rmse_mean_mm']:.1f} mm")
    table = pd.DataFrame(rows).sort_values("rmse_mean_mm")
    subj = pd.DataFrame(per_subject)
    table.to_csv(out_root / "final.csv", index=False)
    subj.to_csv(out_root / "final_per_subject.csv", index=False)
    log_path = out_root / "log.csv"
    log_rows = table.assign(timestamp=stamp)[["timestamp", "model", "tag", "epochs", "rmse_mean_mm",
                                              "rmse_ap_mm", "rmse_ml_mm", "rmse_vert_mm"]]
    log_rows.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    n_looks = pd.read_csv(log_path).timestamp.nunique()
    cfg0 = (model_cfgs or [baseline_cfg])[0]
    notes = subject_notes(cfg0)
    cols = ["model", "tag", "rmse_mean_mm", "rmse_ap_mm", "rmse_ml_mm", "rmse_vert_mm", "exc_rmse_mean_mm"]
    md = ["# Final test (locked subjects)", "",
          f"Run {stamp}. This is look number {n_looks} at the locked subjects (see `log.csv`).", "",
          "Models were retrained on all development subjects; networks for the median best epoch of their LOSO runs, "
          "baselines with the settings their LOSO runs chose.", "",
          "## Locked subjects", *[f"- **{s}**: " + ("; ".join(v) if v else "inside the development range")
                                  for s, v in notes.items()], "",
          "## Pooled over the locked subjects (mm)", md_table(table[cols], index=False, digits=1), "",
          "## Per subject (RMSE, mm)",
          md_table(subj[["model", "tag", "subject", "rmse_mean_mm", "rmse_ap_mm", "rmse_ml_mm", "rmse_vert_mm"]],
                   index=False, digits=1)]
    (out_root / "final.md").write_text("\n".join(md), encoding="utf-8")
    return table
