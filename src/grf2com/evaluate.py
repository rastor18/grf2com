"""Streaming (real-time style) evaluation, metrics, gait-cycle error profiles, CPU latency and result plots.

Whole trials are streamed with windows of eval.window_s ending every eval.hop_s; each window keeps only its newest
hop (the first window keeps everything), so every prediction uses at most window_s of past input, as in real time.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import plotting
from . import preprocess as pp

AXES = ("ap", "ml", "vert")
PROFILE_POINTS = 101


@torch.inference_mode()
def predict_stream(model, x: torch.Tensor, s: torch.Tensor, n: int, k: int, window: int, hop: int,
                   batch: int = 256) -> np.ndarray:
    """Normalized predictions (n, 3) for one trial. x: (n * k, C) on the model's device, s: (S,)."""
    model.eval()
    w = min(window, n)
    ends = sorted(set(range(w, n, hop)) | {n})
    out = np.zeros((n, 3), np.float32)
    prev = 0
    ar = torch.arange(w * k, device=x.device)
    for i in range(0, len(ends), batch):
        chunk = ends[i:i + batch]
        idx = torch.stack([ar + (e - w) * k for e in chunk])
        pred = model(x[idx], s[None].expand(len(chunk), -1))[:, ::k].float().cpu().numpy()
        for row, e in zip(pred, chunk):
            out[prev:e] = row[prev - (e - w):]
            prev = e
    return out


def axis_metrics(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> dict:
    """RMSE / MAE / bias in mm and R² per axis over valid frames (inputs in m)."""
    out = {}
    for a, name in enumerate(AXES):
        e = 1000 * (pred[mask, a] - true[mask, a])
        t = 1000 * true[mask, a]
        out[f"rmse_{name}_mm"] = float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")
        out[f"mae_{name}_mm"] = float(np.mean(np.abs(e))) if e.size else float("nan")
        out[f"bias_{name}_mm"] = float(np.mean(e)) if e.size else float("nan")
        out[f"r2_{name}"] = float(1 - np.sum(e ** 2) / np.sum((t - t.mean()) ** 2)) if e.size > 1 else float("nan")
    out["rmse_mean_mm"] = float(np.mean([out[f"rmse_{a}_mm"] for a in AXES]))
    return out


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Centered NaN-aware moving average (evaluation only; not causal)."""
    w = max(1, int(w))
    finite = np.isfinite(x)
    c = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(np.where(finite, x, 0), 0)])
    n = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(finite, 0)])
    lo = np.clip(np.arange(len(x)) - w // 2, 0, len(x))
    hi = np.clip(np.arange(len(x)) - w // 2 + w, 0, len(x))
    cnt = n[hi] - n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt >= w / 2, (c[hi] - c[lo]) / cnt, np.nan)


def stride_frames(events: dict) -> int:
    hs = events.get("HS_left", np.array([]))
    d = np.diff(hs)
    d = d[(d > 0.6 * pp.COM_RATE) & (d < 2.0 * pp.COM_RATE)]
    return int(np.median(d)) if len(d) else int(1.1 * pp.COM_RATE)


def excursion_errors(pred: np.ndarray, true: np.ndarray, mask: np.ndarray, w: int) -> np.ndarray:
    """Errors (m) after removing a one-stride moving average from both curves: 'how the CoM moves', not 'where'."""
    p = np.where(mask[:, None], pred, np.nan)
    t = np.where(mask[:, None], true, np.nan)
    return (p - moving_average(p, w)) - (t - moving_average(t, w))


def gait_profile(err_mm: np.ndarray, mask: np.ndarray, hs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sum of squared error and counts on a 0-100 % gait cycle (left heel strike to left heel strike)."""
    sse = np.zeros((PROFILE_POINTS, 3))
    cnt = np.zeros(PROFILE_POINTS)
    phase = np.linspace(0, 1, PROFILE_POINTS)
    for a, b in zip(hs[:-1], hs[1:]):
        if not (0.6 * pp.COM_RATE < b - a < 2.0 * pp.COM_RATE) or not mask[a:b + 1].all():
            continue
        f = np.arange(a, b + 1)
        for ax in range(3):
            sse[:, ax] += np.interp(a + phase * (b - a), f, err_mm[a:b + 1, ax]) ** 2
        cnt += 1
    return sse, cnt


def evaluate(model, tset, fb, cfg: dict, full: bool = True) -> dict:
    """Stream every trial of a TensorSet and score it (see score())."""
    e = cfg["eval"]
    window, hop = int(round(e["window_s"] * pp.COM_RATE)), int(round(e["hop_s"] * pp.COM_RATE))
    preds = [fb.denormalize(predict_stream(model, tset.features(i), tset.s[i], t.n_frames, tset.k, window, hop,
                                           int(e.get("batch_size", 256))), t.subject)
             for i, t in enumerate(tset.trials)]
    return score(tset.trials, preds, full)


def score(trials, preds: list[np.ndarray], full: bool = True) -> dict:
    """Metrics for predictions (n, 3) in m, one per trial. Returns pooled metrics, and with full=True also
    per-trial metrics, per-subject metrics, the gait-cycle error profile and the predictions."""
    out_preds, rows = {}, []
    sse, cnt = np.zeros((PROFILE_POINTS, 3)), np.zeros(PROFILE_POINTS)
    pooled = {"pred": [], "true": [], "mask": [], "exc": []}
    for t, p in zip(trials, preds):
        true = np.nan_to_num(t.y.astype(np.float64))
        pooled["pred"].append(p)
        pooled["true"].append(true)
        pooled["mask"].append(t.mask)
        if not full:
            continue
        exc = excursion_errors(p, np.where(t.mask[:, None], true, np.nan), t.mask, stride_frames(t.events))
        pooled["exc"].append(exc)
        row = {"subject": t.subject, "trial": t.trial, "n_valid": int(t.mask.sum()),
               **axis_metrics(p, true, t.mask)}
        for a, name in enumerate(AXES):
            row[f"exc_rmse_{name}_mm"] = float(1000 * np.sqrt(np.nanmean(exc[:, a] ** 2)))
        rows.append(row)
        s_, c_ = gait_profile(1000 * (p - true), t.mask, t.events.get("HS_left", np.array([], int)))
        sse += s_
        cnt += c_
        out_preds[t.name] = {"pred": p.astype(np.float32), "true": t.y, "mask": t.mask}
    P, T, M = (np.concatenate(pooled[key]) for key in ("pred", "true", "mask"))
    out = {"pooled": axis_metrics(P, T, M)}
    if not full:
        return out
    exc = np.concatenate(pooled["exc"])
    for a, name in enumerate(AXES):
        out["pooled"][f"exc_rmse_{name}_mm"] = float(1000 * np.sqrt(np.nanmean(exc[:, a] ** 2)))
    out["pooled"]["exc_rmse_mean_mm"] = float(np.mean([out["pooled"][f"exc_rmse_{a}_mm"] for a in AXES]))
    per_trial = pd.DataFrame(rows)
    subj = []
    for s in sorted({t.subject for t in trials}):
        sel = [i for i, t in enumerate(trials) if t.subject == s]
        subj.append({"subject": s, **axis_metrics(np.concatenate([pooled["pred"][i] for i in sel]),
                                                    np.concatenate([pooled["true"][i] for i in sel]),
                                                    np.concatenate([pooled["mask"][i] for i in sel]))})
    with np.errstate(invalid="ignore"):
        profile = np.sqrt(sse / cnt[:, None])
    out.update(per_trial=per_trial, per_subject=pd.DataFrame(subj), profile=profile, n_strides=int(cnt.max()),
               preds=out_preds)
    return out


def cpu_latency_ms(model, in_ch: int, subj_dim: int, window_samples: int, runs: int = 30) -> float:
    """Median time of one batch-1 forward over a full window on the CPU (the per-sample cost when streaming)."""
    m = copy.deepcopy(model).cpu().eval()
    x = torch.randn(1, window_samples, in_ch)
    s = torch.randn(1, subj_dim)
    times = []
    with torch.inference_mode():
        for i in range(runs + 3):
            t0 = time.perf_counter()
            m(x, s)
            if i >= 3:
                times.append(time.perf_counter() - t0)
    return float(1000 * np.median(times))


def save_plots(result: dict, history: pd.DataFrame | None, out_dir: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    plotting.apply_style()
    labels = ("AP", "ML", "vertical")
    if history is not None and len(history):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        axes[0].plot(history.epoch, history.train_loss, color=plotting.SERIES[0])
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("train loss (normalized MSE)")
        axes[0].set_title("Training loss")
        if "val_rmse_mm" in history:
            axes[1].plot(history.epoch, history.val_rmse_mm, color=plotting.SERIES[0])
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("validation RMSE, mean of axes (mm)")
        axes[1].set_title("Validation")
        plotting.save(fig, out_dir / "curves.png")

    name, d = next(iter(result["preds"].items()))
    n = len(d["true"])
    a = max(0, n // 2 - 10 * pp.COM_RATE)
    sl = slice(a, min(n, a + 20 * pp.COM_RATE))
    t = np.arange(sl.start, sl.stop) / pp.COM_RATE
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for i, ax in enumerate(axes):
        true = np.where(d["mask"][sl], d["true"][sl, i], np.nan) * 1000
        ax.plot(t, true, color=plotting.SERIES[0], lw=1.2, label="measured")
        ax.plot(t, d["pred"][sl, i] * 1000, color=plotting.SERIES[1], lw=1.2, label="predicted")
        ax.set_ylabel(f"CoM {labels[i]} (mm)")
    axes[0].legend(loc="upper right", ncol=2)
    axes[0].set_title(f"{title}: {name}, 20 s, streamed")
    axes[-1].set_xlabel("time (s)")
    plotting.save(fig, out_dir / "trace.png")

    fig, ax = plt.subplots(figsize=(6, 3.4))
    pct = np.linspace(0, 100, PROFILE_POINTS)
    for i in range(3):
        ax.plot(pct, result["profile"][:, i], color=plotting.SERIES[i], label=labels[i])
    ax.set_xlabel("gait cycle (% from left heel strike)")
    ax.set_ylabel("RMS error (mm)")
    ax.set_title(f"{title}: error over the gait cycle ({result['n_strides']} strides)")
    ax.legend(loc="upper right", ncol=3)
    plotting.save(fig, out_dir / "gait_profile.png")
