"""Final test on the locked subjects (data.final_test_subjects). Run ONCE, at the very end, and only with --confirm.

Retrains the chosen configs on every development trial (for the median best epoch of their LOSO runs) and the
baselines (with the settings their LOSO runs chose), then scores the locked subjects. Results go to results/final/;
every run is appended to results/final/log.csv.

Example: scripts/final_test.py --configs configs/models/tcn.yaml configs/models/gru.yaml --tag full_s0 --confirm
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grf2com.config import add_config_args, load_config, project_path  # noqa: E402
from grf2com.device import pick_device  # noqa: E402
from grf2com.final import run_final  # noqa: E402
from grf2com.tables import final_test_subjects  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser, model_config=False)
    parser.add_argument("--configs", nargs="*", default=[], help="model configs to retrain and test")
    parser.add_argument("--tag", default="full_s0", help="development run tag that supplies the epoch counts")
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--baseline-tag", default="full", help="development baseline tag that supplies settings")
    parser.add_argument("--allow-incomplete", action="store_true", help="allow configs whose LOSO runs are partial")
    parser.add_argument("--device")
    parser.add_argument("--confirm", action="store_true", help="required: this looks at the locked subjects")
    args = parser.parse_args()
    base = load_config(None, args.set)
    locked = final_test_subjects(base)
    if not args.confirm:
        print(f"Refusing to run: the final test scores the locked subjects {locked}, which should be looked at "
              "once, at the very end. Re-run with --confirm when the comparison is settled.", file=sys.stderr)
        sys.exit(2)
    cfgs = [load_config(c, args.set) for c in args.configs]
    device = pick_device(args.device or base["train"]["device"])
    table = run_final(cfgs, None if args.no_baselines else base, args.tag, args.baseline_tag, device,
                      project_path(base, "results_dir") / "final", args.allow_incomplete)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
