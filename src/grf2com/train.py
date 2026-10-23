"""Training for one fold: AdamW + cosine schedule, early stopping on validation RMSE, streamed test evaluation.

Writes runs/<model>/<tag>/<fold>/: config.json, history.csv, best.pt, metrics.json, per_trial.csv, preds.npz, plots.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import preprocess as pp
from .config import config_hash, project_path
from .data import FeatureBuilder, Fold, TensorSet, load_trials, rate_mult
from .device import describe
from .evaluate import cpu_latency_ms, evaluate, save_plots
from .models import build_model, count_parameters
from .tables import load_infotable


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_dir(cfg: dict, tag: str, fold: str) -> Path:
    return project_path(cfg, "runs_dir") / cfg["model"]["name"] / tag / fold


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def masked_mse(pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return (((pred - y) ** 2).mean(-1) * m).sum() / m.sum().clamp(min=1)


def save_checkpoint(path: Path, model, fb: FeatureBuilder, cfg: dict, extra: dict | None = None) -> None:
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "features": fb.state,
                "config": cfg, "in_ch": fb.n_inputs, "subj_dim": fb.n_subject, **(extra or {})}, path)


def load_checkpoint(path: Path, device: torch.device | str = "cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg, ck["in_ch"], ck["subj_dim"])
    model.load_state_dict(ck["model"])
    fb = FeatureBuilder(cfg).load_state(ck["features"])
    return model.to(device), fb, cfg, ck


def train_fold(cfg: dict, fold: Fold, out_dir: Path, device: torch.device, dev_mode: bool = False,
               log=print) -> dict:
    t_start = time.perf_counter()
    tcfg = cfg["train"]
    seed = int(tcfg["seed"])
    set_seed(seed)
    k = rate_mult(cfg)
    cdir = pp.cache_dir(cfg)
    if not (cdir / "summary.csv").exists():
        raise FileNotFoundError(f"no cache at {cdir}; run scripts/preprocess.py with the same preprocess config")
    info = load_infotable(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_trials = load_trials(cfg, fold.train, cdir)
    val_trials = load_trials(cfg, fold.val, cdir)
    test_trials = load_trials(cfg, fold.test, cdir)
    fb = FeatureBuilder(cfg).fit(train_trials, info)
    train_set = TensorSet(train_trials, fb, info, k, device)
    val_set = TensorSet(val_trials, fb, info, k, device) if val_trials else None

    model = build_model(cfg, fb.n_inputs, fb.n_subject).to(device)
    n_params = count_parameters(model)
    rf = model.receptive_field()
    epochs = int(tcfg["dev_epochs"] if dev_mode else tcfg["epochs"])
    steps = int(tcfg["dev_steps_per_epoch"] if dev_mode else tcfg["steps_per_epoch"])
    patience = int(tcfg["patience"])
    length = int(round(cfg["data"]["window_s"] * pp.COM_RATE))
    warmup = int(round(cfg["data"]["warmup_s"] * pp.COM_RATE))
    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps)
    rng = np.random.default_rng(seed)

    meta = {"fold": fold.name, "train_subjects": fold.subjects("train"), "val_subjects": fold.subjects("val"),
            "test_subjects": fold.subjects("test"), "n_train_trials": len(train_trials), "cache": cdir.name,
            "config_hash": config_hash({k_: v for k_, v in cfg.items() if k_ != "config_file"}),
            "device": describe(device), "n_params": n_params, "receptive_field_samples": rf,
            "receptive_field_s": rf / (pp.COM_RATE * k) if rf else None, "dev_mode": dev_mode}
    (out_dir / "config.json").write_text(json.dumps(_jsonable({"config": cfg, "meta": meta}), indent=1))
    log(f"[{fold.name}] {cfg['model']['name']}: {n_params:,} params"
        + (f", receptive field {rf} samples ({meta['receptive_field_s']:.2f} s)" if rf else "")
        + f", train {meta['train_subjects']}, val {meta['val_subjects']}, test {meta['test_subjects']}, "
          f"{describe(device)}")

    best, best_epoch, bad, history = math.inf, 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        total = torch.zeros((), device=device)
        for _ in range(steps):
            x, s, y, m = train_set.sample(int(tcfg["batch_size"]), length, warmup, rng)
            loss = masked_mse(model(x, s)[:, ::k], y, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
            opt.step()
            sched.step()
            total += loss.detach()
        row = {"epoch": epoch, "train_loss": float(total) / steps, "lr": sched.get_last_lr()[0],
               "train_s": time.perf_counter() - t0}
        if val_set is not None:
            row["val_rmse_mm"] = evaluate(model, val_set, fb, cfg, full=False)["pooled"]["rmse_mean_mm"]
            score = row["val_rmse_mm"]
        else:
            score = -epoch   # no validation (final test): keep the last epoch
        history.append(row)
        improved = score < best
        if improved:
            best, best_epoch, bad = score, epoch, 0
            save_checkpoint(out_dir / "best.pt", model, fb, cfg, {"epoch": epoch, "meta": meta})
        else:
            bad += 1
        log(f"[{fold.name}] epoch {epoch:3d}  loss {row['train_loss']:.4f}"
            + (f"  val {row['val_rmse_mm']:.1f} mm" if "val_rmse_mm" in row else "")
            + f"  ({row['train_s']:.1f} s){' *' if improved else ''}")
        if val_set is not None and bad >= patience:
            break
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    model, fb, _, _ = load_checkpoint(out_dir / "best.pt", device)
    summary = {**meta, "best_epoch": best_epoch, "epochs_run": len(history),
               "best_val_rmse_mm": best if val_set is not None else None}
    if test_trials:
        test_set = TensorSet(test_trials, fb, info, k, device)
        res = evaluate(model, test_set, fb, cfg, full=True)
        res["per_trial"].to_csv(out_dir / "per_trial.csv", index=False)
        np.savez_compressed(out_dir / "preds.npz", **{f"{n}__{key}": v for n, d in res["preds"].items()
                                                       for key, v in d.items()})
        save_plots(res, pd.DataFrame(history), out_dir, f"{cfg['model']['name']} / {fold.name}")
        summary["test"] = res["pooled"]
        summary["per_subject"] = res["per_subject"].to_dict(orient="records")
        summary["gait_profile_rms_mm"] = np.nan_to_num(res["profile"]).round(2).tolist()
    window = int(round(cfg["eval"]["window_s"] * pp.COM_RATE)) * k
    summary["cpu_latency_ms"] = cpu_latency_ms(model, fb.n_inputs, fb.n_subject, window,
                                               int(cfg["eval"].get("latency_runs", 30)))
    summary["wall_s"] = time.perf_counter() - t_start
    (out_dir / "metrics.json").write_text(json.dumps(_jsonable(summary), indent=1))
    if "test" in summary:
        t = summary["test"]
        log(f"[{fold.name}] test RMSE AP {t['rmse_ap_mm']:.1f}, ML {t['rmse_ml_mm']:.1f}, vert {t['rmse_vert_mm']:.1f} mm"
            f" (excursion {t['exc_rmse_ap_mm']:.1f} / {t['exc_rmse_ml_mm']:.1f} / {t['exc_rmse_vert_mm']:.1f}); "
            f"latency {summary['cpu_latency_ms']:.2f} ms; {summary['wall_s']:.0f} s")
    return summary
