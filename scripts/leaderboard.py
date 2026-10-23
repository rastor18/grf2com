"""Collect runs/<model>/<tag>/<fold>/metrics.json into results/leaderboard.{csv,md}: mean ± sd across development
folds per (model, tag), with parameter count and CPU latency. Dev and smoke runs are skipped unless --all."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from grf2com import transitions  # noqa: E402
from grf2com.config import add_config_args, config_from_args, project_path  # noqa: E402
from grf2com.data import make_folds  # noqa: E402
from grf2com.reporting import md_table  # noqa: E402
from grf2com.tables import final_test_subjects  # noqa: E402

METRICS = ["rmse_mean_mm", "rmse_ap_mm", "rmse_ml_mm", "rmse_vert_mm", "exc_rmse_mean_mm",
           "r2_ap", "r2_ml", "r2_vert"]
SKIP_TAGS = ("dev", "smoke")


def collect(runs_dir: Path, include_all: bool) -> pd.DataFrame:
    rows = []
    for path in runs_dir.glob("*/*/*/metrics.json"):
        model, tag, fold = path.parts[-4:-1]
        if not include_all and any(s in tag for s in SKIP_TAGS):
            continue
        m = json.loads(path.read_text())
        if "test" not in m:
            continue
        rows.append({"model": model, "tag": tag, "fold": fold, **{k: m["test"].get(k) for k in METRICS},
                     "n_params": m.get("n_params"), "cpu_latency_ms": m.get("cpu_latency_ms"),
                     "best_epoch": m.get("best_epoch"), "test_subjects": ",".join(m.get("test_subjects", []))})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser, model_config=False)
    parser.add_argument("--all", action="store_true", help="include dev / smoke runs")
    args = parser.parse_args()
    cfg = config_from_args(args)
    df = collect(project_path(cfg, "runs_dir"), args.all)
    out = project_path(cfg, "results_dir")
    out.mkdir(parents=True, exist_ok=True)
    if df.empty:
        sys.exit("no finished runs found")
    locked = set(final_test_subjects(cfg))
    leaked = df[df.test_subjects.str.split(",").apply(lambda s: bool(set(s) & locked))]
    if len(leaked):
        sys.exit(f"locked final-test subjects found in development runs: {leaked[['model', 'tag', 'fold']]}")
    n_folds = len(make_folds(cfg))
    g = df.groupby(["model", "tag"])
    board = g[METRICS].mean().add_suffix("")
    sd = g[METRICS].std().add_suffix("_sd")
    board = pd.concat([board, sd], axis=1)
    board["folds"] = g.size()
    board["complete"] = board.folds >= n_folds
    board["n_params"] = g.n_params.first()
    board["cpu_latency_ms"] = g.cpu_latency_ms.median()
    board["median_best_epoch"] = g.best_epoch.median()
    board = board.sort_values("rmse_mean_mm").reset_index()
    board.to_csv(out / "leaderboard.csv", index=False)
    df.to_csv(out / "leaderboard_folds.csv", index=False)

    def pm(r, k, d=1):
        return f"{r[k]:.{d}f} ± {r[k + '_sd']:.{d}f}" if np.isfinite(r[k + "_sd"]) else f"{r[k]:.{d}f}"

    show = pd.DataFrame({
        "model": board.model, "tag": board.tag, "folds": board.folds.astype(str) + "/" + str(n_folds),
        "RMSE mean (mm)": [pm(r, "rmse_mean_mm") for _, r in board.iterrows()],
        "AP": [pm(r, "rmse_ap_mm") for _, r in board.iterrows()],
        "ML": [pm(r, "rmse_ml_mm") for _, r in board.iterrows()],
        "vertical": [pm(r, "rmse_vert_mm") for _, r in board.iterrows()],
        "excursion (mm)": [pm(r, "exc_rmse_mean_mm") for _, r in board.iterrows()],
        "params": board.n_params.map(lambda v: f"{int(v):,}" if pd.notna(v) else ""),
        "CPU latency (ms)": board.cpu_latency_ms.map(lambda v: f"{v:.2f}" if pd.notna(v) else ""),
    })
    md = ["# Leaderboard (development subjects, leave one subject out)", "",
          f"Mean ± sd across folds of the streamed test RMSE. {n_folds} development folds; the locked final-test "
          f"subjects ({', '.join(sorted(locked))}) never appear here. Excursion = RMSE after removing a one-stride "
          "moving average from both curves. Latency = one batch-1 forward over a 4 s window on the CPU.", "",
          md_table(show, index=False)]
    tdf = transitions.collect(project_path(cfg, "runs_dir"))
    if len(tdf):
        tb = transitions.aggregate(tdf)
        tshow = pd.DataFrame({
            "model": tb.model, "tag": tb.tag, "events": tb.n_events.astype(int),
            "steady AP": [pm(r, "steady_rmse_ap_mm") for _, r in tb.iterrows()],
            "pulse AP": [pm(r, "pulse_rmse_ap_mm") for _, r in tb.iterrows()],
            "pulse/steady AP": [pm(r, "pulse_ratio_ap", 2) for _, r in tb.iterrows()],
            "steady AP shape": [pm(r, "steady_exc_ap_mm") for _, r in tb.iterrows()],
            "pulse AP shape": [pm(r, "pulse_exc_ap_mm") for _, r in tb.iterrows()],
            "50/50 score": [pm(r, "score_50_50_mm") for _, r in tb.iterrows()],
        })
        md += ["", "## Steady walking vs speed-change pulses", "",
               "From `scripts/transition_report.py` (details in `results/transitions.md`). Pulse = onset to 1 s after "
               "the speed settles; steady = the constant-speed stretch before the pulse phase. Shape = error after "
               "removing a one-stride moving average. 50/50 score = mean of steady and pulse RMSE over all axes.", "",
               md_table(tshow, index=False)]
    (out / "leaderboard.md").write_text("\n".join(md), encoding="utf-8")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
