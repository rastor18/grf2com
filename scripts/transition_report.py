"""Score every finished development run on treadmill speed transitions (see grf2com/transitions.py).

Reads runs/<model>/<tag>/<fold>/preds.npz, writes transitions.json next to it, and writes
results/transitions.{csv,md} and results/transitions/*.png. No retraining.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from grf2com import plotting  # noqa: E402
from grf2com import transitions as tr  # noqa: E402
from grf2com.config import add_config_args, config_from_args, project_path  # noqa: E402
from grf2com.data import load_trials  # noqa: E402
from grf2com.reporting import md_table  # noqa: E402
from grf2com.tables import final_test_subjects  # noqa: E402
from grf2com.train import _jsonable  # noqa: E402

REFERENCE = ("tcn", "full_s0")
PLOT_TAGS = {"full_s0", "full"}


def load_preds(path: Path) -> dict:
    out = {}
    with np.load(path) as z:
        for key in z.files:
            name, field = key.split("__")
            out.setdefault(name, {})[field] = z[key]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser, model_config=False)
    parser.add_argument("--force", action="store_true", help="rescore runs that already have transitions.json")
    args = parser.parse_args()
    cfg = config_from_args(args)
    runs_dir = project_path(cfg, "runs_dir")
    locked = set(final_test_subjects(cfg))
    cache: dict = {}
    for path in sorted(runs_dir.glob("*/*/*/preds.npz")):
        model, tag, fold = path.parts[-4:-1]
        if any(s in tag for s in ("dev", "smoke")):
            continue
        out = path.with_name("transitions.json")
        if out.exists() and not args.force:
            continue
        preds = load_preds(path)
        names = sorted(preds)
        if {n.split("_")[0] for n in names} & locked:
            sys.exit(f"locked subject in {path}")
        missing = [n for n in names if n not in cache]
        if missing:
            trials = load_trials(cfg, [tuple(n.split("_")) for n in missing])
            ev, st = tr.events_for_trials(cfg, trials)
            for n in missing:
                cache[n] = (ev[n], st[n])
        events = {n: cache[n][0] for n in names}
        strides = {n: cache[n][1] for n in names}
        res = tr.score_run(cfg, preds, events, strides)
        for by in ("direction", "phase"):
            for value, rm in tr.breakdown(preds, events, strides, by).items():
                for k, ax in enumerate(("ap", "ml", "vert")):
                    res[f"trans_rmse_{ax}_mm_{value}"] = rm[k]
        out.write_text(json.dumps(_jsonable(res)))
        print(f"{model}/{tag}/{fold}: steady AP {res['steady_rmse_ap_mm']:.1f}, transition AP "
              f"{res['trans_rmse_ap_mm']:.1f} mm ({res['n_events']} events)", flush=True)

    df = tr.collect(runs_dir)
    results = project_path(cfg, "results_dir")
    df.to_csv(results / "transitions_folds.csv", index=False)
    board = tr.aggregate(df)
    board.to_csv(results / "transitions.csv", index=False)

    # paired per-fold comparison against the reference on transition AP and on the 50/50 score
    ref = df[(df.model == REFERENCE[0]) & (df.tag == REFERENCE[1])].set_index("fold")
    paired = []
    for (m, t), g in df.groupby(["model", "tag"]):
        if (m, t) == REFERENCE:
            continue
        g = g.set_index("fold")
        common = g.index.intersection(ref.index)
        for col in ("pulse_rmse_ap_mm", "pulse_exc_ap_mm", "score_50_50_mm"):
            d = g.loc[common, col] - ref.loc[common, col]
            paired.append({"model": m, "tag": t, "metric": col, "diff_mean": d.mean(), "diff_sd": d.std(),
                           "worse_than_ref": f"{int((d > 0).sum())}/{len(d)}"})
    paired = pd.DataFrame(paired)
    paired.to_csv(results / "transitions_paired.csv", index=False)

    def pm(r, k, digits=1):
        sd = r.get(f"{k}_sd", np.nan)
        return f"{r[k]:.{digits}f} ± {sd:.{digits}f}" if np.isfinite(sd) else f"{r[k]:.{digits}f}"

    show = pd.DataFrame([{
        "model": r.model, "tag": r.tag, "folds": int(r.folds), "events": int(r.n_events),
        "steady AP": pm(r, "steady_rmse_ap_mm"), "pulse AP": pm(r, "pulse_rmse_ap_mm"),
        "pulse/steady AP": pm(r, "pulse_ratio_ap", 2), "pulse AP shape": pm(r, "pulse_exc_ap_mm"),
        "long-window AP": pm(r, "trans_rmse_ap_mm"), "steady mean": pm(r, "steady_rmse_mean_mm"),
        "pulse mean": pm(r, "pulse_rmse_mean_mm"), "50/50 score": pm(r, "score_50_50_mm"),
        "peak AP": pm(r, "peak_ap_mm_median"), "recovery AP (s)": pm(r, "recovery_ap_s_median", 2),
        "recovered": f"{r.recovered_ap_frac:.0%}"} for _, r in board.iterrows()])
    split = pd.DataFrame([{
        "model": r.model, "tag": r.tag, "slow pulses": f"{r.trans_rmse_ap_mm_slow:.1f}",
        "fast pulses": f"{r.trans_rmse_ap_mm_fast:.1f}", "early half": f"{r.trans_rmse_ap_mm_early:.1f}",
        "late half": f"{r.trans_rmse_ap_mm_late:.1f}"} for _, r in board.iterrows()])
    pshow = paired[paired.metric == "pulse_rmse_ap_mm"].sort_values("diff_mean")
    md = ["# Speed transitions (development subjects, leave one subject out)", "",
          "Events are ±0.5 m/s treadmill pulses (1.25 → 0.75 → 1.25 and 1.25 → 1.75 → 1.25 m/s). Transition window: "
          "pulse onset to 1 s after the speed settles ('pulse', the headline), and a long window from 2 s before onset to "
          "8 s after the speed settles, cut at the next pulse ('long-window'). Steady: the constant 1.25 m/s "
          "stretch before the pulse phase (earlier in the trial, so time-in-trial is not controlled). All pulses have "
          "the same size, so there is no breakdown by size. Mean ± sd across folds; RMSE in mm. Peak = median over "
          "events of the largest 0.5 s RMS error; recovery = median time after the speed settles until the 0.5 s RMS "
          f"error is back within {tr.RECOVERY_X}× steady ('recovered' = share of events where that happens within the "
          "window). 50/50 score = mean of steady and during-pulse RMSE (all axes). 'pulse AP shape' = AP error after "
          "removing a one-stride moving average (how the CoM moves, not where).", "",
          md_table(show, index=False), "",
          "## During-pulse AP RMSE by pulse direction and by half of the pulse phase (mm, mean over folds)",
          md_table(split, index=False), "",
          f"## Paired per-fold difference in during-pulse AP RMSE vs {REFERENCE[0]} / {REFERENCE[1]} (mm; > 0 = worse)",
          md_table(pshow[["model", "tag", "diff_mean", "diff_sd", "worse_than_ref"]], index=False, digits=1), "",
          f"## Paired per-fold difference in during-pulse AP shape error vs {REFERENCE[0]} / {REFERENCE[1]} (mm)",
          md_table(paired[paired.metric == "pulse_exc_ap_mm"].sort_values("diff_mean")[
              ["model", "tag", "diff_mean", "diff_sd", "worse_than_ref"]], index=False, digits=1), "",
          "Plot: `transitions/curves.png` (mean absolute error vs time since pulse onset)."]
    (results / "transitions.md").write_text("\n".join(md), encoding="utf-8")

    # event-locked curves for the main runs
    import matplotlib.pyplot as plt
    plotting.apply_style()
    curves = {}
    for p in runs_dir.glob("*/*/*/transitions.json"):
        model, tag, fold = p.parts[-4:-1]
        if tag not in PLOT_TAGS:
            continue
        d = json.loads(p.read_text())
        for direction in ("slow", "fast"):
            if f"curve_{direction}" in d:
                curves.setdefault((model, direction), []).append(np.array(d[f"curve_{direction}"], float))
                lags = np.array(d["curve_lags_s"])
    order = [m for m in ("tcn", "transformer", "gru", "lstm", "mlp", "ridge", "kalman", "song_kf")
             if (m, "slow") in curves]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
    for i, direction in enumerate(("slow", "fast")):
        for j, ax_name in enumerate(("AP", "ML", "vertical")):
            ax = axes[i, j]
            for c, m in enumerate(order):
                y = np.nanmean(np.stack(curves[(m, direction)]), axis=0)[:, j]
                ax.plot(lags, y, color=plotting.SERIES[c], lw=1.4, label=m)
            ax.axvline(0, color=plotting.TEXT_2, lw=0.8)
            ax.axvline(3.45, color=plotting.TEXT_2, lw=0.8, ls="--")
            ax.set_title(f"{ax_name}, {direction} pulses")
            if j == 0:
                ax.set_ylabel("mean |error| (mm)")
            if i == 1:
                ax.set_xlabel("time since pulse onset (s); dashed = speed settles")
    axes[0, 0].legend(ncol=2, loc="upper left")
    plotting.save(fig, results / "transitions" / "curves.png")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
