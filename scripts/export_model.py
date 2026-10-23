"""Train a deployment model on every development trial and write a bundle for the real-time runtime.

The locked final-test subjects are never loaded. Epochs default to the median best epoch of the config's development
LOSO runs (--tag), as in the final test; pass --epochs to override.

Example:
  scripts/export_model.py --config configs/models/tcn.yaml --tag full_s0 --name tcn_v1
  -> deploy/models/tcn_v1/{model.pt, bundle.json}
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grf2com.bundle import loso_summary, write_bundle  # noqa: E402
from grf2com.config import ROOT, add_config_args, config_from_args  # noqa: E402
from grf2com.data import Fold, _trials_by_subject  # noqa: E402
from grf2com.device import pick_device  # noqa: E402
from grf2com.tables import development_subjects, final_test_subjects  # noqa: E402
from grf2com.train import run_dir, train_fold  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    parser.add_argument("--tag", default="full_s0", help="development LOSO tag for the epoch count and the summary")
    parser.add_argument("--epochs", type=int, help="override the epoch count")
    parser.add_argument("--name", required=True, help="bundle name, e.g. tcn_v1")
    parser.add_argument("--out", default=str(ROOT / "deploy" / "models"))
    parser.add_argument("--device")
    args = parser.parse_args()
    cfg = config_from_args(args)
    loso, median_epoch = loso_summary(cfg, cfg["model"]["name"], args.tag)
    epochs = args.epochs or median_epoch
    if not epochs:
        sys.exit(f"no development runs for {cfg['model']['name']}/{args.tag}; pass --epochs")
    cfg["train"]["epochs"] = int(epochs)
    by_subject = _trials_by_subject(cfg)
    dev = development_subjects(cfg, by_subject)
    assert not set(dev) & set(final_test_subjects(cfg))
    fold = Fold("deploy", [(s, t) for s in dev for t in by_subject[s]], [], [])
    work = run_dir(cfg, f"deploy_{args.name}", "all")
    print(f"training {cfg['model']['name']} on {len(fold.train)} development trials ({', '.join(dev)}) for {epochs} "
          f"epochs", flush=True)
    summary = train_fold(cfg, fold, work, pick_device(args.device or cfg["train"]["device"]))
    out = write_bundle(Path(args.out) / args.name, work / "best.pt", cfg, summary, loso)
    print(f"bundle written: {out}")


if __name__ == "__main__":
    main()
