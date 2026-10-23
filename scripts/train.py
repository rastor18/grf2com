"""Train one model config on development folds (LOSO by default).

Example: scripts/train.py --config configs/models/tcn.yaml --folds AE03 --dev
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grf2com.config import add_config_args, config_from_args  # noqa: E402
from grf2com.data import make_folds  # noqa: E402
from grf2com.device import pick_device  # noqa: E402
from grf2com.train import run_dir, train_fold  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    parser.add_argument("--folds", nargs="*", help="fold names to run (LOSO: test subject, e.g. AE03); default all")
    parser.add_argument("--dev", action="store_true", help="quick run: few trials per subject, few short epochs")
    parser.add_argument("--tag", help="run tag (default: 'dev' with --dev, else 'default')")
    parser.add_argument("--device", help="cpu, xpu, cuda, mps (default: train.device)")
    parser.add_argument("--force", action="store_true", help="rerun folds that already have metrics.json")
    args = parser.parse_args()
    cfg = config_from_args(args)
    device = pick_device(args.device or cfg["train"]["device"])
    tag = args.tag or ("dev" if args.dev else "default")
    for fold in make_folds(cfg, dev_mode=args.dev, only=args.folds):
        out = run_dir(cfg, tag, fold.name)
        if (out / "metrics.json").exists() and not args.force:
            print(f"[{fold.name}] already done: {out}")
            continue
        summary = train_fold(cfg, fold, out, device, dev_mode=args.dev)
        if "test" in summary:
            print(json.dumps({k: round(v, 2) for k, v in summary["test"].items()}))


if __name__ == "__main__":
    main()
