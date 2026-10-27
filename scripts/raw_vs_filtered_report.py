"""Raw vs filtered force input for the TCN: results/raw_vs_filtered.{csv,md}.

Arms (development LOSO, 8 folds):
  filtered          tcn/full_s0              CSV forces (MATLAB filtfilt 10 Hz, zero phase: not causal)
  raw               tcn/raw_s{0,1,2}         unfiltered C3D forces, only our causal 25 Hz low-pass
  raw_pre10         tcn/raw_pre10_s{0,1,2}   raw + causal 10 Hz Butterworth x 2 passes (the live emulation)
  filtered->pre10   tcn/full_s0_on_raw_pre10 the CSV-trained model fed raw + causal 10 Hz x 2 (today's deployment)
  filtered->raw     tcn/full_s0_on_raw       the CSV-trained model fed raw forces

Per fold: RMSE (mean, AP, ML, vertical), stride excursion AP, AP movement shape steady / pulse and pulse AP RMSE
(transitions.json, from scripts/transition_report.py) and the self-pace command error vs the mocap oracle
(results/selfpace/replay_steps.csv, from scripts/selfpace_replay.py, if those nets are in it). Seeds are averaged per
fold first; the seed spread is the sd over seeds of the 8-fold mean. Paired differences are per fold.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grf2com import selfpace as sp  # noqa: E402
from grf2com.config import add_config_args, config_from_args, project_path  # noqa: E402
from grf2com.reporting import md_table  # noqa: E402

ARMS = {
    "filtered": ["full_s0", "full_s1", "full_s2"],
    "raw": ["raw_s0", "raw_s1", "raw_s2"],
    "raw_pre10": ["raw_pre10_s0", "raw_pre10_s1", "raw_pre10_s2"],
    "filtered->pre10": ["full_s0_on_raw_pre10"],
    "filtered->raw": ["full_s0_on_raw"],
}
# (reference, other, what the difference other - reference measures)
PAIRS = [
    ("filtered", "raw_pre10", "cost of losing the zero-phase filter's future information"),
    ("raw_pre10", "raw", "effect of dropping the causal 10 Hz smoothing"),
    ("filtered", "raw", "filtered vs fully raw"),
    ("raw_pre10", "filtered->pre10", "cost of not retraining (today's deployment vs a matched model)"),
    ("raw", "filtered->raw", "cost of not retraining, no prefilter"),
]
METRICS = {
    "rmse_mean_mm": "RMSE mean", "rmse_ap_mm": "AP", "rmse_ml_mm": "ML", "rmse_vert_mm": "vert",
    "exc_rmse_ap_mm": "excursion AP", "steady_exc_ap_mm": "AP shape steady", "pulse_exc_ap_mm": "AP shape pulse",
    "pulse_rmse_ap_mm": "pulse AP RMSE", "cmd_rmse_mps": "self-pace cmd (m/s)",
}


def fold_table(runs: Path, model: str, steps: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    for arm, tags in ARMS.items():
        for tag in tags:
            for path in sorted((runs / model / tag).glob("*/metrics.json")):
                m = json.loads(path.read_text())
                if "test" not in m:
                    continue
                row = {"arm": arm, "tag": tag, "fold": path.parent.name,
                       **{k: m["test"].get(k) for k in METRICS if k in m["test"]}}
                tr = path.with_name("transitions.json")
                if tr.exists():
                    t = json.loads(tr.read_text())
                    row.update({k: t.get(k) for k in ("steady_exc_ap_mm", "pulse_exc_ap_mm", "pulse_rmse_ap_mm")})
                rows.append(row)
    df = pd.DataFrame(rows)
    if steps is not None and len(df):
        nets = [f"{model}@{t}" for t in df.tag.unique() if f"{model}@{t}_v" in steps]
        if nets:
            cmd = sp.summarize(steps, nets, by="subject")   # replay_steps.csv is already scored (sp.score)
            cmd["tag"] = cmd.estimator.str.split("@").str[1]
            df = df.merge(cmd[["subject", "tag", "cmd_rmse_mps"]].rename(columns={"subject": "fold"}),
                          on=["tag", "fold"], how="left")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser, model_config=False)
    parser.add_argument("--model", default="tcn")
    args = parser.parse_args()
    cfg = config_from_args(args)
    results = project_path(cfg, "results_dir")
    steps_path = results / "selfpace" / "replay_steps.csv"
    steps = pd.read_csv(steps_path) if steps_path.exists() else None
    df = fold_table(project_path(cfg, "runs_dir"), args.model, steps)
    if df.empty:
        sys.exit("no runs found")
    df.to_csv(results / "raw_vs_filtered.csv", index=False)
    metrics = [k for k in METRICS if k in df]

    per_fold = df.groupby(["arm", "fold"])[metrics].mean()            # seeds averaged per fold
    seed_means = df.groupby(["arm", "tag"])[metrics].mean()            # 8-fold mean per seed
    summary = []
    for arm in ARMS:
        if arm not in per_fold.index.get_level_values(0):
            continue
        sm = seed_means.loc[arm]
        row = {"arm": arm, "seeds": len(sm), "folds": len(per_fold.loc[arm])}
        for k in metrics:
            row[METRICS[k]] = f"{sm[k].mean():.{3 if k == 'cmd_rmse_mps' else 1}f}" + (
                f" ± {sm[k].std():.{3 if k == 'cmd_rmse_mps' else 1}f}" if len(sm) > 1 else "")
        summary.append(row)
    summary = pd.DataFrame(summary).set_index("arm")

    paired = []
    for ref, other, meaning in PAIRS:
        if ref not in per_fold.index.get_level_values(0) or other not in per_fold.index.get_level_values(0):
            continue
        a, b = per_fold.loc[ref], per_fold.loc[other]
        common = a.index.intersection(b.index)
        for k in metrics:
            d = (b.loc[common, k] - a.loc[common, k]).dropna()
            if d.empty:
                continue
            spreads = [seed_means.loc[x][k].std() for x in (ref, other) if len(seed_means.loc[x]) > 1]
            spread = float(np.nanmax(spreads)) if spreads else float("nan")
            paired.append({"comparison": f"{other} - {ref}", "meaning": meaning, "metric": METRICS[k],
                           "diff_mean": d.mean(), "diff_sd_folds": d.std(), "worse_folds": f"{int((d > 0).sum())}/{len(d)}",
                           "seed_spread": spread, "diff_per_spread": d.mean() / spread if spread else float("nan")})
    paired = pd.DataFrame(paired)
    paired.to_csv(results / "raw_vs_filtered_paired.csv", index=False)

    md = ["# Raw vs filtered force input (TCN, development LOSO)", "",
          "Arms: `filtered` = CSV forces (zero-phase filtfilt 10 Hz, not causal); `raw` = unfiltered C3D forces with only "
          "the causal 25 Hz low-pass; `raw_pre10` = raw + causal 10 Hz Butterworth x 2 passes (the live emulation of "
          "the CSV filter); `filtered->X` = the CSV-trained model evaluated on input X without retraining.", "",
          "Mean over folds (± sd over seeds; seeds are averaged per fold first). mm unless noted.", "",
          md_table(summary, 1), "",
          "## Paired per fold", "",
          "diff = other - reference (positive = other is worse); worse_folds = folds where the other arm is worse; "
          "seed_spread = the larger sd over seeds of the 8-fold mean; diff_per_spread = how many seed spreads the "
          "difference is.", ""]
    for comp, g in (paired.groupby("comparison", sort=False) if len(paired) else []):
        md += [f"### {comp}: {g.meaning.iloc[0]}", "",
               md_table(g[["metric", "diff_mean", "diff_sd_folds", "worse_folds", "seed_spread", "diff_per_spread"]],
                        3, index=False), ""]
    md += ["## Per fold (seed mean)", "", md_table(per_fold.reset_index().pivot(index="fold", columns="arm",
                                                                                   values="rmse_mean_mm"), 1), ""]
    (results / "raw_vs_filtered.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:9]))


if __name__ == "__main__":
    main()
