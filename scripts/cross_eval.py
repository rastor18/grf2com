"""Evaluate trained runs on differently preprocessed inputs, without retraining (train/deploy mismatch).

Each fold's best.pt (weights + fitted FeatureBuilder state, so the training normalization is kept) is streamed over
the same test trials taken from the cache of a sweep ablation, e.g. the CSV-trained TCN on raw forces with the live
causal prefilter. Output: runs/<model>/<tag>_on_<ablation>/<fold>/ (config.json, metrics.json, per_trial.csv,
preds.npz), so leaderboard.py, transition_report.py and selfpace_replay.py pick it up like any run.

Example: scripts/cross_eval.py --run tcn/full_s0 --ablations raw_pre10 raw
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sweep import ABLATIONS  # noqa: E402

from grf2com import preprocess as pp  # noqa: E402
from grf2com.config import ROOT, deep_merge, load_config, parse_override, project_path  # noqa: E402
from grf2com.data import TensorSet, load_trials, rate_mult  # noqa: E402
from grf2com.device import pick_device  # noqa: E402
from grf2com.evaluate import evaluate  # noqa: E402
from grf2com.tables import final_test_subjects, load_infotable  # noqa: E402
from grf2com.train import _jsonable, load_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="tcn/full_s0", help="model/tag of the trained runs")
    parser.add_argument("--ablations", nargs="+", default=["raw_pre10", "raw"], choices=list(ABLATIONS),
                        help="input preprocessing to evaluate on (the sweep ablation's preprocess / realtime keys)")
    parser.add_argument("--folds", nargs="*")
    parser.add_argument("--device")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    model_name, tag = args.run.split("/")
    src = ROOT / "runs" / model_name / tag
    folds = sorted(p.parent.name for p in src.glob("*/best.pt") if not args.folds or p.parent.name in args.folds)
    if not folds:
        sys.exit(f"no best.pt under {src}")
    for ablation in args.ablations:
        for fold in folds:
            model, fb, cfg, _ = load_checkpoint(src / fold / "best.pt")
            cfg["paths"] = {**load_config()["paths"], **cfg["paths"]}   # runs from before raw_grf_dir existed
            # only the input side changes: preprocess.* and realtime.* overrides of the ablation
            over = [o for o in ABLATIONS[ablation] if o.split(".")[0] in ("preprocess", "realtime")]
            for item in over:
                cfg = deep_merge(cfg, parse_override(item))
            out = project_path(cfg, "runs_dir") / model_name / f"{tag}_on_{ablation}" / fold
            if (out / "metrics.json").exists() and not args.force:
                continue
            if not (pp.cache_dir(cfg) / "summary.csv").exists():
                pp.run(cfg)
            device = pick_device(args.device or cfg["train"]["device"])
            model = model.to(device)
            metrics = json.loads((src / fold / "metrics.json").read_text())
            test = [tuple(n.split("_")) for n in metrics_trials(src / fold)]
            if {s for s, _ in test} & set(final_test_subjects(cfg)):
                sys.exit(f"locked subject in {src / fold}")
            info = load_infotable(cfg)
            tset = TensorSet(load_trials(cfg, test), fb, info, rate_mult(cfg), device)
            res = evaluate(model, tset, fb, cfg, full=True)
            out.mkdir(parents=True, exist_ok=True)
            res["per_trial"].to_csv(out / "per_trial.csv", index=False)
            np.savez_compressed(out / "preds.npz", **{f"{n}__{key}": v for n, d in res["preds"].items()
                                                       for key, v in d.items()})
            keep = ("fold", "train_subjects", "val_subjects", "test_subjects", "n_params", "best_epoch",
                    "cpu_latency_ms", "receptive_field_s")
            summary = {**{k: metrics[k] for k in keep if k in metrics}, "trained_run": f"{model_name}/{tag}",
                       "eval_ablation": ablation, "cache": pp.cache_dir(cfg).name, "test": res["pooled"],
                       "per_subject": res["per_subject"].to_dict(orient="records")}
            (out / "metrics.json").write_text(json.dumps(_jsonable(summary), indent=1))
            (out / "config.json").write_text(json.dumps(_jsonable({"config": cfg, "meta": summary}), indent=1))
            t = res["pooled"]
            print(f"[{fold}] {model_name}/{tag} on {ablation}: RMSE mean {t['rmse_mean_mm']:.1f} mm "
                  f"(AP {t['rmse_ap_mm']:.1f}, ML {t['rmse_ml_mm']:.1f}, vert {t['rmse_vert_mm']:.1f})", flush=True)


def metrics_trials(run: Path) -> list[str]:
    """Test trial names of a finished run (from its per-trial table)."""
    df = pd.read_csv(run / "per_trial.csv")
    return [f"{s}_{t}" for s, t in zip(df.subject, df.trial)]


if __name__ == "__main__":
    main()
