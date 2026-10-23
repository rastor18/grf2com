"""Live shadow-mode GUI: stream the Bertec force data (or a synthetic test stream) into one or more model bundles,
watch the CoM in 3D and as time series, and record logs together with a QTM capture. It never commands the treadmill.

  scripts/live_gui.py                       # bundles from deploy/models/, logs to deploy/logs/
  scripts/live_gui.py --set gui.treadmill.center=[0.1,0.0]

See DEPLOY.md ("Live GUI") and src/grf2com/gui.py.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grf2com.config import add_config_args, config_from_args, resolve_file  # noqa: E402
from grf2com.gui import main  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="deploy/models", help="folder of model bundles")
    parser.add_argument("--logs", default="deploy/logs", help="folder for the recorded logs")
    add_config_args(parser, model_config=False)
    args = parser.parse_args()
    main(config_from_args(args), resolve_file(args.models), resolve_file(args.logs))
