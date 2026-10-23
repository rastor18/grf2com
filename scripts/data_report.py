"""Data report on the development subjects (the locked final-test subjects are left out).

Writes results/data_report.{csv,md} and results/data_report/*.png:
plate offsets and drift, mean Fz / body weight, force <-> mocap lag, CoM ranges and NaN %, how well stride-averaged
COP predicts stride-averaged absolute CoM position across subjects, and a belt-speed check from stance-phase COP.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from grf2com import plotting  # noqa: E402
from grf2com import preprocess as pp  # noqa: E402
from grf2com.config import add_config_args, config_from_args, project_path  # noqa: E402
from grf2com.reporting import md_table  # noqa: E402
from grf2com.tables import development_subjects, final_test_subjects, load_trial_index  # noqa: E402

CH = {c: i for i, c in enumerate(pp.CHANNELS)}
AXES = {"ap": (0, "Fy"), "ml": (1, "Fx"), "vert": (2, "Fz")}   # target column, force pair
COP = ["COPx_l", "COPx_r", "COPy_l", "COPy_r"]
MAX_LAG = 6
STRIDE_FRAMES = 66        # ~1.1 s, one stride, for stride averages
SUBSAMPLE = 15
MARGIN = 3                # frames trimmed from each end of a stance for the belt check


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Centered moving average (analysis only, not causal); NaN unless all w samples are finite."""
    x = np.asarray(x, float)
    finite = np.isfinite(x)
    c = np.concatenate([[0.0], np.cumsum(np.where(finite, x, 0.0))])
    n = np.concatenate([[0], np.cumsum(finite)])
    out = np.full(len(x), np.nan)
    h = w // 2
    if len(x) < w:
        return out
    s, cnt = c[w:] - c[:-w], n[w:] - n[:-w]
    out[h:h + len(s)] = np.where(cnt == w, s / w, np.nan)
    return out


def shifted_corr(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    """corr(a[j], b[j + lag]) over finite pairs."""
    if lag >= 0:
        a2, b2 = a[:len(a) - lag], b[lag:]
    else:
        a2, b2 = a[-lag:], b[:len(b) + lag]
    ok = np.isfinite(a2) & np.isfinite(b2)
    return float(np.corrcoef(a2[ok], b2[ok])[0, 1]) if ok.sum() > 10 else float("nan")


def force_lag(x60: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict:
    """Lag (frames) of the cached force behind the mocap CoM acceleration, per axis; > 0 means force is later."""
    acc = np.full(len(y), np.nan)[:, None].repeat(3, 1)
    ok = mask[2:] & mask[1:-1] & mask[:-2]
    acc[1:-1] = np.where(ok[:, None], y[2:] - 2 * y[1:-1] + y[:-2], np.nan)
    out = {}
    lags = np.arange(-MAX_LAG, MAX_LAG + 1)
    for name, (col, f) in AXES.items():
        force = x60[:, CH[f"{f}_l"]] + x60[:, CH[f"{f}_r"]]
        r = np.array([shifted_corr(acc[:, col], force, int(L)) for L in lags])
        i = int(np.nanargmax(np.abs(r)))
        frac = 0.0
        if 0 < i < len(lags) - 1:   # parabolic refinement on |r|
            a, b, c = np.abs(r[i - 1:i + 2])
            denom = a - 2 * b + c
            frac = 0.5 * (a - c) / denom if denom else 0.0
        out[f"lag_{name}_frames"] = float(lags[i] + frac)
        out[f"lag_{name}_r"] = float(r[i])
    return out


def stride_averaged(x60: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stride-averaged COP features and CoM targets, subsampled."""
    feats = np.stack([moving_average(x60[:, CH[c]], STRIDE_FRAMES) for c in COP], axis=1)
    yv = np.where(mask[:, None], y, np.nan)
    targ = np.stack([moving_average(yv[:, a], STRIDE_FRAMES) for a in range(3)], axis=1)
    ok = np.isfinite(feats).all(1) & np.isfinite(targ).all(1)
    idx = np.flatnonzero(ok)[::SUBSAMPLE]
    return feats[idx], targ[idx]


def lstsq_fit(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(A, Y, rcond=None)[0]


def lstsq_predict(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X]) @ coef


def stances(d: dict, side: str, n: int) -> list[tuple[int, int]]:
    hs, to = d.get(f"ev_HS_{pp.SIDES[side]}", []), d.get(f"ev_TO_{pp.SIDES[side]}", [])
    out = []
    for i, h in enumerate(hs):
        j = np.searchsorted(to, h, side="right")
        if j >= len(to):
            break
        t = int(to[j])
        nxt = hs[i + 1] if i + 1 < len(hs) else n
        if 0.25 * 60 <= t - h <= 1.2 * 60 and t < nxt:
            out.append((int(h) + MARGIN, t - MARGIN))
    return out


def belt_rows(x60: np.ndarray, d: dict, shift: int = 0) -> np.ndarray:
    """Per stance: [dCOPx (mm), dCOPy (mm), v*T (m)], with the speed channel shifted by `shift` frames."""
    n = len(x60)
    speed = x60[:, CH["speed"]]
    if shift:
        speed = np.roll(speed, shift)
    rows = []
    for s in pp.SIDES:
        contact = x60[:, CH[f"contact_{s}"]]
        for a, b in stances(d, s, n):
            if b - a < 6 or b >= n or contact[a] < 0.9 or contact[b] < 0.9:
                continue
            dx = x60[b, CH[f"COPx_{s}"]] - x60[a, CH[f"COPx_{s}"]]
            dy = x60[b, CH[f"COPy_{s}"]] - x60[a, CH[f"COPy_{s}"]]
            rows.append([dx, dy, speed[a:b].mean() * (b - a) / 60])
    return np.array(rows).reshape(-1, 3)


def line_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    slope, icpt = np.polyfit(x[ok], y[ok], 1)
    r2 = 1 - np.var(y[ok] - (slope * x[ok] + icpt)) / np.var(y[ok])
    return float(slope), float(icpt), float(r2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser, model_config=False)
    args = parser.parse_args()
    cfg = config_from_args(args)
    cdir = pp.cache_dir(cfg)
    if not (cdir / "summary.csv").exists():
        sys.exit(f"no cache at {cdir}; run scripts/preprocess.py first")
    out_dir = project_path(cfg, "results_dir")
    plot_dir = out_dir / "data_report"
    plotting.apply_style()

    k = int(cfg["preprocess"]["grf_rate"]) // pp.COM_RATE
    index = load_trial_index(cfg)
    dev = development_subjects(cfg, index.subject)
    locked = final_test_subjects(cfg)
    trials = [(s, t) for s, t in zip(index.subject, index.trial) if s in dev]
    summary = pp.load_summary(cdir).set_index(["subject", "trial"])

    rows, cop_sets, belt, examples, loaded = [], {}, [], {}, []
    for s, t in trials:
        d = pp.load_trial(cdir, s, t)
        x60, y, mask = d["x"][::k].astype(float), d["y"].astype(float), d["mask"]
        loaded.append((x60, d))
        sm = summary.loc[(s, t)]
        row = {"subject": s, "trial": t, "n_frames": len(y), "valid_frac": sm.valid_frac,
               "com_nan_frac": sm.com_nan_frac, "fz_total_bw_mean": sm.fz_total_bw_mean}
        for q in ("Fx", "Fy", "Fz", "My"):
            for side in pp.SIDES:
                row[f"off_{q}_{side}"] = sm[f"off_{q}_{side}"]
        row["max_drift_F_n"] = sm.filter(like="drift_F").abs().max()
        row["max_drift_M"] = sm.filter(like="drift_M").abs().max()
        row.update(force_lag(x60, y, mask))
        for a, name in enumerate(("ap", "ml", "vert")):
            yv = y[mask, a]
            row[f"com_{name}_min_mm"] = 1000 * yv.min()
            row[f"com_{name}_max_mm"] = 1000 * yv.max()
            row[f"com_{name}_std_mm"] = 1000 * yv.std()
        X, Y = stride_averaged(x60, y, mask)
        cop_sets[(s, t)] = (X, Y)
        coef = lstsq_fit(X, Y[:, :2])
        res = Y[:, :2] - lstsq_predict(coef, X)
        for a, name in enumerate(("ap", "ml")):
            row[f"cop_fit_{name}_r2"] = 1 - res[:, a].var() / Y[:, a].var()
            row[f"cop_fit_{name}_rmse_mm"] = 1000 * np.sqrt((res[:, a] ** 2).mean())
        b = belt_rows(x60, d)
        belt.append(pd.DataFrame(b, columns=["dcopx_mm", "dcopy_mm", "vT_m"]).assign(subject=s, trial=t))
        for c, col in ((0, "x"), (1, "y")):
            if len(b) > 10:
                row[f"belt_slope_cop{col}"], _, row[f"belt_r2_cop{col}"] = line_fit(b[:, 2], b[:, c])
        rows.append(row)
        if s not in examples:
            examples[s] = (t, x60, y, mask)
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "data_report.csv", index=False)
    belt = pd.concat(belt, ignore_index=True)

    # COP -> CoM across subjects: fit on the other development subjects, predict the held-out one.
    loso = []
    for s in dev:
        tr = [v for (ss, _), v in cop_sets.items() if ss != s]
        te = [v for (ss, _), v in cop_sets.items() if ss == s]
        Xtr, Ytr = np.concatenate([v[0] for v in tr]), np.concatenate([v[1] for v in tr])
        Xte, Yte = np.concatenate([v[0] for v in te]), np.concatenate([v[1] for v in te])
        pred = lstsq_predict(lstsq_fit(Xtr, Ytr), Xte)
        for a, name in enumerate(("ap", "ml", "vert")):
            loso.append({"subject": s, "axis": name,
                         "rmse_mm": 1000 * np.sqrt(((pred[:, a] - Yte[:, a]) ** 2).mean()),
                         "mean_baseline_rmse_mm": 1000 * np.sqrt(((Ytr[:, a].mean() - Yte[:, a]) ** 2).mean()),
                         "pred": pred[:, a], "true": Yte[:, a]})
    loso_df = pd.DataFrame(loso)
    pooled_coef = lstsq_fit(np.concatenate([v[0] for v in cop_sets.values()]),
                            np.concatenate([v[1] for v in cop_sets.values()]))

    # Belt check pooled over development trials, with a timing search on the speed channel.
    fits = {c: line_fit(belt.vT_m.to_numpy(), belt[f"dcop{c}_mm"].to_numpy()) for c in ("x", "y")}
    belt_axis = max(fits, key=lambda c: fits[c][2])
    shifts = list(range(-120, 121, 15))
    shift_r2 = []
    for sh in shifts:
        bs = np.concatenate([belt_rows(x60, d, sh) for x60, d in loaded])
        shift_r2.append(line_fit(bs[:, 2], bs[:, 0 if belt_axis == "x" else 1])[2])

    # Plots
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.2))
    for i, name in enumerate(("ap", "ml", "vert")):
        v = df[f"lag_{name}_frames"].to_numpy() * 1000 / pp.COM_RATE
        jit = np.random.default_rng(i).uniform(-0.15, 0.15, len(v))
        ax.scatter(np.full(len(v), i) + jit, v, s=14, color=plotting.SERIES[0], alpha=0.7, linewidths=0)
    ax.set_xticks([0, 1, 2], ["AP (Fy)", "ML (Fx)", "vertical (Fz)"])
    ax.set_ylabel("force lag behind mocap (ms)")
    ax.set_title("Force ↔ mocap lag per trial (cached, filtered force)")
    ax.axhline(0, color=plotting.TEXT_2, lw=0.8)
    plotting.save(fig, plot_dir / "lag.png")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for ax, name in zip(axes, ("ap", "ml")):
        sub = loso_df[loso_df.axis == name]
        p, tr = 1000 * np.concatenate(sub.pred.to_list()), 1000 * np.concatenate(sub.true.to_list())
        ax.scatter(tr, p, s=4, alpha=0.25, color=plotting.SERIES[0], linewidths=0)
        lo, hi = np.percentile(np.r_[tr, p], [0.5, 99.5])
        ax.plot([lo, hi], [lo, hi], color=plotting.TEXT_2, lw=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"measured stride-avg CoM {name.upper()} (mm)")
        ax.set_ylabel("predicted from COP (mm)")
        ax.set_title(f"{name.upper()}: COP → CoM, leave one subject out")
    plotting.save(fig, plot_dir / "cop_to_com.png")

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    yb = belt[f"dcop{belt_axis}_mm"].to_numpy()
    ax.scatter(belt.vT_m, yb, s=4, alpha=0.2, color=plotting.SERIES[0], linewidths=0)
    slope, icpt, r2 = fits[belt_axis]
    xs = np.linspace(belt.vT_m.min(), belt.vT_m.max(), 2)
    ax.plot(xs, slope * xs + icpt, color=plotting.SERIES[1], lw=2)
    ax.set_xlabel("belt speed × stance time (m)")
    ax.set_ylabel(f"COP{belt_axis} change over stance (mm)")
    ax.set_title(f"Belt check: slope {slope:.0f} mm/m, R² {r2:.2f}")
    plotting.save(fig, plot_dir / "belt_check.png")

    fig, ax = plt.subplots(figsize=(6, 3.2))
    per_subj = df.groupby("subject").fz_total_bw_mean.mean()
    ax.bar(per_subj.index, per_subj.to_numpy(), width=0.6, color=plotting.SERIES[0])
    ax.axhline(1, color=plotting.TEXT_2, lw=0.8)
    ax.set_ylim(0.9, max(1.25, per_subj.max() + 0.02))
    ax.set_ylabel("mean total Fz / (infotable mass · g)")
    ax.set_title("Mean vertical force vs. body weight")
    plotting.save(fig, plot_dir / "fz_bw.png")

    s0 = dev[0]
    t0, x60, y, mask = examples[s0]
    a, b = 60 * 60, 60 * 60 + 10 * 60
    tt = np.arange(a, b) / pp.COM_RATE
    fig, axes = plt.subplots(6, 1, figsize=(8, 9), sharex=True)
    for ax, (ch, label) in zip(axes[:3], (("Fz", "Fz (N)"), ("COPy", "COPy (mm)"), ("Fy", "Fy (N)"))):
        for i, side in enumerate(pp.SIDES):
            ax.plot(tt, x60[a:b, CH[f"{ch}_{side}"]], color=plotting.SERIES[i], lw=1, label=pp.SIDES[side])
        ax.set_ylabel(label)
    axes[0].legend(loc="upper right", ncol=2)
    for ax, i, label in zip(axes[3:], range(3), ("CoM AP (mm)", "CoM ML (mm)", "CoM vertical (mm)")):
        ax.plot(tt, 1000 * np.where(mask[a:b], y[a:b, i], np.nan), color=plotting.SERIES[0], lw=1)
        ax.set_ylabel(label)
    axes[-1].set_xlabel("time (s)")
    axes[0].set_title(f"{s0} {t0}, 10 s of cached inputs and targets")
    plotting.save(fig, plot_dir / "example_trace.png")

    # Markdown
    ms = 1000 / pp.COM_RATE
    lag_txt = ", ".join(
        f"{n.upper()} {df[f'lag_{n}_frames'].median():+.2f} frames ({df[f'lag_{n}_frames'].median() * ms:+.0f} ms, "
        f"IQR {df[f'lag_{n}_frames'].quantile(.25) * ms:+.0f} to {df[f'lag_{n}_frames'].quantile(.75) * ms:+.0f} ms, "
        f"median |r| {df[f'lag_{n}_r'].abs().median():.2f})"
        for n in ("ap", "ml", "vert"))
    loso_tab = loso_df.groupby("axis")[["rmse_mm", "mean_baseline_rmse_mm"]].mean().loc[["ap", "ml", "vert"]]
    subj = df.groupby("subject").agg(
        trials=("trial", "size"), valid=("valid_frac", "mean"), fz_bw=("fz_total_bw_mean", "mean"),
        lag_ap_ms=("lag_ap_frames", lambda v: v.median() * ms), lag_ml_ms=("lag_ml_frames", lambda v: v.median() * ms),
        lag_vert_ms=("lag_vert_frames", lambda v: v.median() * ms),
        ap_min_mm=("com_ap_min_mm", "min"), ap_max_mm=("com_ap_max_mm", "max"),
        ml_min_mm=("com_ml_min_mm", "min"), ml_max_mm=("com_ml_max_mm", "max"),
        vert_mean_mm=("com_vert_min_mm", "mean"), cop_r2_ap=("cop_fit_ap_r2", "median"),
        cop_r2_ml=("cop_fit_ml_r2", "median"))
    loso_subj = loso_df.pivot(index="subject", columns="axis", values="rmse_mm")[["ap", "ml", "vert"]]
    best_shift = shifts[int(np.argmax(shift_r2))]
    extra_kg = (per_subj - 1) * summary.mass_kg.groupby(level=0).first().loc[per_subj.index]
    md = [
        "# Data report (development subjects)",
        "",
        f"Cache `{cdir.relative_to(project_path(cfg, 'cache_dir').parent).as_posix()}`, input rate "
        f"{cfg['preprocess']['grf_rate']} Hz. Development subjects: {', '.join(dev)} ({len(trials)} trials). "
        f"The locked final-test subjects ({', '.join(locked)}) are not in this report.",
        "",
        "## Plates",
        f"- Tare offsets (median while unloaded): Fz {df[['off_Fz_l', 'off_Fz_r']].min().min():.0f} to "
        f"{df[['off_Fz_l', 'off_Fz_r']].max().max():.0f} N, Fy {df[['off_Fy_l', 'off_Fy_r']].min().min():.0f} to "
        f"{df[['off_Fy_l', 'off_Fy_r']].max().max():.0f} N, Fx {df[['off_Fx_l', 'off_Fx_r']].min().min():.0f} to "
        f"{df[['off_Fx_l', 'off_Fx_r']].max().max():.0f} N.",
        f"- Offset drift between the first and last minute: at most {df.max_drift_F_n.max():.2f} N (forces) and "
        f"{df.max_drift_M.max():.2f} (moments) in any trial, so one tare per trial is enough.",
        f"- Mean total Fz / (infotable mass · g) is {df.fz_total_bw_mean.min():.3f} to {df.fz_total_bw_mean.max():.3f} "
        f"(per subject {per_subj.min():.3f} to {per_subj.max():.3f}). Over a walking trial the mean vertical force equals "
        f"the supported weight, so subjects carry about {extra_kg.min():.1f}–{extra_kg.max():.1f} kg more than the "
        "infotable mass (exo, shoes, markers?). `force_per_bw` divides by infotable mass, so "
        "normalized forces keep a subject-specific offset.",
        "",
        "## Force ↔ mocap timing",
        f"- Lag of the cached (causally filtered) force behind the mocap CoM acceleration: {lag_txt}. "
        "Positive means the force arrives later. The 25 Hz 2nd-order causal Butterworth alone adds about 9 ms.",
        "",
        "## COP → absolute CoM position (stride averages)",
        f"- Per trial, a linear map from the four stride-averaged COP channels explains median R² "
        f"{df.cop_fit_ap_r2.median():.2f} (AP) and {df.cop_fit_ml_r2.median():.2f} (ML) of the stride-averaged CoM.",
        f"- Across subjects (fit on 7 development subjects, predict the 8th): RMSE "
        f"{loso_tab.loc['ap', 'rmse_mm']:.1f} mm AP, {loso_tab.loc['ml', 'rmse_mm']:.1f} mm ML, "
        f"{loso_tab.loc['vert', 'rmse_mm']:.1f} mm vertical, against "
        f"{loso_tab.loc['ap', 'mean_baseline_rmse_mm']:.1f} / {loso_tab.loc['ml', 'mean_baseline_rmse_mm']:.1f} / "
        f"{loso_tab.loc['vert', 'mean_baseline_rmse_mm']:.1f} mm for the training mean. "
        "If the COP map beats the mean on AP and ML, absolute position is recoverable from plate COP across subjects.",
        f"- Pooled coefficients (m per mm, COPx_l, COPx_r, COPy_l, COPy_r): AP "
        f"{np.array2string(pooled_coef[1:, 0], precision=5)}, ML {np.array2string(pooled_coef[1:, 1], precision=5)}.",
        "",
        "## Belt speed check",
        f"- Change in COP over a stance vs belt speed × stance time, pooled over {len(belt)} stances: COPx slope "
        f"{fits['x'][0]:.0f} mm/m (R² {fits['x'][2]:.2f}), COPy slope {fits['y'][0]:.0f} mm/m (R² {fits['y'][2]:.2f}). "
        f"A slope near ±1000 mm/m on one axis confirms m/s units and identifies the belt axis (COP{belt_axis}); the sign "
        "gives the belt direction in plate coordinates.",
        f"- Timing: shifting the speed channel by -2 to +2 s changes R² from {min(shift_r2):.3f} to "
        f"{max(shift_r2):.3f}; best shift {best_shift / pp.COM_RATE:+.2f} s (R² at 0 s: "
        f"{shift_r2[shifts.index(0)]:.3f}). Speed changes are slow ramps, so this only rules out large timing errors.",
        "",
        "## Valid frames",
        f"- Valid fraction per trial: min {df.valid_frac.min():.3f}, median {df.valid_frac.median():.3f}. "
        f"Trials below 0.97: {', '.join(f'{r.subject} {r.trial} ({r.valid_frac:.3f})' for r in df[df.valid_frac < 0.97].itertuples()) or 'none'}.",
        "",
        "## Per subject",
        md_table(subj.assign(extra_kg=extra_kg), digits=3),
        "",
        "## COP → CoM, leave one subject out (RMSE, mm)",
        md_table(loso_subj, digits=1),
        "",
        "Plots: `data_report/lag.png`, `cop_to_com.png`, `belt_check.png`, `fz_bw.png`, `example_trace.png`. "
        "Per-trial numbers: `data_report.csv`.",
    ]
    (out_dir / "data_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out_dir / 'data_report.md'} and {plot_dir}")


if __name__ == "__main__":
    main()
