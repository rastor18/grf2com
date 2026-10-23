"""Run models x ablations x folds x seeds on the development subjects (resumable: finished runs are skipped).

Each run is runs/<model>/<ablation>_s<seed>/<fold>/. With --baselines, the classical baselines are run for each
ablation too (under the same tag). Ablations that change preprocessing build their cache first.

Examples:
  scripts/sweep.py                                   # all models, 'full' ablation, all folds, seed 0
  scripts/sweep.py --ablations full source_cop no_subject no_speed --baselines
  scripts/sweep.py --models tcn gru --folds AE03 --seeds 0 1 2
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grf2com import preprocess as pp  # noqa: E402
from grf2com.baselines import BASELINES  # noqa: E402
from grf2com.config import ROOT, add_config_args, load_config, project_path  # noqa: E402
from grf2com.data import is_single_belt, make_folds  # noqa: E402
from grf2com.device import pick_device  # noqa: E402
from grf2com.models import MODELS  # noqa: E402
from grf2com.tables import load_infotable  # noqa: E402
from grf2com.train import run_dir, train_fold  # noqa: E402

ABLATIONS = {
    "full": [],
    "source_cop": ["preprocess.cop_from_moments=false"],            # exported COP instead of recomputed
    "no_subject": ["data.subject_features=[]"],
    "no_speed": ["data.features=[forces,cop,moments,contact]"],
    "forces_only": ["data.features=[forces,contact,speed]"],
    "forces_cop": ["data.features=[forces,cop,contact,speed]"],
    "vert_resid": ["target.vertical=anthro_residual"],          # vertical = anthropometric estimate + network
    "vert_absolute": ["target.vertical=absolute"],              # network predicts absolute height (old default)
    "source_cop_vabs": ["preprocess.cop_from_moments=false", "target.vertical=absolute"],
    "single_belt": ["data.features=[total_force,single_cop,single_moments,speed]"],   # regular treadmill
    "plus_labkf": ["data.features=[forces,cop,moments,contact,speed,lab_kf]"],      # + lab KF state
    "plus_kalman": ["data.features=[forces,cop,moments,contact,speed,kalman_state]"],   # + our Kalman's estimate
    "plus_both": ["data.features=[forces,cop,moments,contact,speed,lab_kf,kalman_state]"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser, model_config=False)
    parser.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--ablations", nargs="*", default=["full"], choices=list(ABLATIONS))
    parser.add_argument("--folds", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--baselines", action="store_true", help="also run the classical baselines per ablation")
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--device")
    args = parser.parse_args()

    if args.baselines:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from run_baselines import run_one  # noqa: E402
    t_all = time.perf_counter()
    for ablation in args.ablations:
        for seed in args.seeds:
            tag = f"{ablation}_s{seed}" + ("_dev" if args.dev else "")
            for model in args.models:
                cfg = load_config(ROOT / "configs" / "models" / f"{model}.yaml",
                                  [*ABLATIONS[ablation], *args.set, f"train.seed={seed}"])
                if not (pp.cache_dir(cfg) / "summary.csv").exists():
                    pp.run(cfg)
                device = pick_device(args.device or cfg["train"]["device"])
                for fold in make_folds(cfg, dev_mode=args.dev, only=args.folds):
                    out = run_dir(cfg, tag, fold.name)
                    if (out / "metrics.json").exists():
                        continue
                    train_fold(cfg, fold, out, device, dev_mode=args.dev)
            if args.baselines and seed == args.seeds[0]:
                cfg = load_config(None, [*ABLATIONS[ablation], *args.set])
                if not (pp.cache_dir(cfg) / "summary.csv").exists():
                    pp.run(cfg)
                info = load_infotable(cfg)
                for fold in make_folds(cfg, dev_mode=args.dev, only=args.folds):
                    for name in BASELINES:
                        if is_single_belt(cfg) and getattr(BASELINES[name], "split_belt_only", False):
                            continue
                        out = project_path(cfg, "runs_dir") / name / f"{ablation}" / fold.name
                        if not (out / "metrics.json").exists():
                            s = run_one(cfg, name, fold, info, pp.cache_dir(cfg), out)
                            print(f"[{fold.name}] {name} {ablation}: RMSE mean {s['test']['rmse_mean_mm']:.1f} mm",
                                  flush=True)
    print(f"sweep done in {(time.perf_counter() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
