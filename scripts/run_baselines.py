"""Run the classical baselines (mean, ridge, Kalman) on every development fold.

Results go to runs/<baseline>/<tag>/<fold>/metrics.json in the same format as the networks, so the leaderboard
includes them.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from grf2com import preprocess as pp  # noqa: E402
from grf2com.baselines import BASELINES  # noqa: E402
from grf2com.config import add_config_args, config_from_args, project_path  # noqa: E402
from grf2com.data import is_single_belt, load_trials, make_folds  # noqa: E402
from grf2com.evaluate import score  # noqa: E402
from grf2com.tables import load_infotable  # noqa: E402
from grf2com.train import _jsonable  # noqa: E402


def run_one(cfg, name, fold, info, cdir, out_dir) -> dict:
    t0 = time.perf_counter()
    train, val, test = (load_trials(cfg, getattr(fold, p), cdir) for p in ("train", "val", "test"))
    model = BASELINES[name](cfg).fit(train, val, info)
    t_fit = time.perf_counter() - t0
    res = score(test, [model.predict(t, info) for t in test], full=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    res["per_trial"].to_csv(out_dir / "per_trial.csv", index=False)
    np.savez_compressed(out_dir / "preds.npz", **{f"{n}__{key}": v for n, d in res["preds"].items()
                                                   for key, v in d.items()})
    summary = {"fold": fold.name, "train_subjects": fold.subjects("train"), "val_subjects": fold.subjects("val"),
               "test_subjects": fold.subjects("test"), "n_params": model.n_params, "cache": cdir.name,
               "test": res["pooled"], "per_subject": res["per_subject"].to_dict(orient="records"),
               "gait_profile_rms_mm": np.nan_to_num(res["profile"]).round(2).tolist(), "fit_s": t_fit,
               "cpu_latency_ms": None}
    for attr in ("alpha", "qr", "gain"):
        if hasattr(model, attr):
            summary[attr] = getattr(model, attr)
    (out_dir / "metrics.json").write_text(json.dumps(_jsonable({k: (v if not isinstance(v, dict) else
                                                                    {str(a): b for a, b in v.items()})
                                                                for k, v in summary.items()}), indent=1))
    (out_dir / "config.json").write_text(json.dumps(_jsonable({"config": cfg, "meta": {"baseline": name}}), indent=1))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser, model_config=False)
    parser.add_argument("--baselines", nargs="*", default=list(BASELINES), choices=list(BASELINES))
    parser.add_argument("--folds", nargs="*")
    parser.add_argument("--tag", default="default")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = config_from_args(args)
    cdir = pp.cache_dir(cfg)
    info = load_infotable(cfg)
    for fold in make_folds(cfg, only=args.folds):
        for name in args.baselines:
            if is_single_belt(cfg) and getattr(BASELINES[name], "split_belt_only", False):
                continue
            out = project_path(cfg, "runs_dir") / name / args.tag / fold.name
            if (out / "metrics.json").exists() and not args.force:
                continue
            t0 = time.perf_counter()
            s = run_one(cfg, name, fold, info, cdir, out)
            t = s["test"]
            print(f"[{fold.name}] {name:6s} RMSE AP {t['rmse_ap_mm']:6.1f}  ML {t['rmse_ml_mm']:5.1f}  "
                  f"vert {t['rmse_vert_mm']:5.1f} mm  (excursion {t['exc_rmse_ap_mm']:.1f} / {t['exc_rmse_ml_mm']:.1f}"
                  f" / {t['exc_rmse_vert_mm']:.1f})  {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
