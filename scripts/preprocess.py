"""Build cache/<hash>/<subject>_<trial>.npz from the raw CSVs (raw data is only read)."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grf2com import preprocess  # noqa: E402
from grf2com.config import add_config_args, config_from_args  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser, model_config=False)
    parser.add_argument("--trials", nargs="*", help="subset, e.g. AE01_T1 AE02_T3 (default: all trials)")
    parser.add_argument("--workers", type=int, help="parallel workers (default: preprocess.workers)")
    parser.add_argument("--force", action="store_true", help="redo trials that are already cached")
    args = parser.parse_args()
    cfg = config_from_args(args)
    trials = [tuple(name.split("_")) for name in args.trials] if args.trials else None
    t0 = time.perf_counter()
    cdir = preprocess.run(cfg, trials, args.workers, args.force)
    print(f"done in {time.perf_counter() - t0:.0f} s: {cdir}")


if __name__ == "__main__":
    main()
