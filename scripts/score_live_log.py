"""Score a live (or replay) shadow-mode log against mocap CoM recorded in the same session.

The CoM file is a QTM/MATLAB export in the training format (*_CoM.csv: frame, time, CoM_Pos_1..3). The two clocks
are aligned by cross-correlating the slow AP drift (coarse) and then the vertical oscillation (fine), unless
--offset-s is given (CoM time = log time + offset). Reports RMSE / bias per axis and the movement-shape error.

Example: scripts/score_live_log.py --log deploy/logs/20261024_101500_bertec.csv --com path/to/AE12_T1_CoM.csv
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import signal  # noqa: E402

from grf2com.evaluate import AXES, axis_metrics, excursion_errors  # noqa: E402

FS = 60.0


def best_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> int:
    """Lag (samples) that best aligns b to a: a[i] ~ b[i + lag]."""
    a, b = np.nan_to_num(a - np.nanmean(a)), np.nan_to_num(b - np.nanmean(b))
    xc = signal.correlate(b, a, mode="full")
    lags = signal.correlation_lags(len(b), len(a), mode="full")
    ok = np.abs(lags) <= max_lag
    return int(lags[ok][np.argmax(xc[ok])])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", required=True)
    parser.add_argument("--com", required=True)
    parser.add_argument("--offset-s", type=float, help="CoM time = log time + offset (skip automatic alignment)")
    parser.add_argument("--max-lag-s", type=float, default=60.0)
    parser.add_argument("--stride-s", type=float, default=1.1, help="stride time for the movement-shape metric")
    args = parser.parse_args()
    log = pd.read_csv(args.log)
    com = pd.read_csv(args.com)
    t = log.t.to_numpy() - log.t.iloc[0]
    grid = np.arange(0, t[-1], 1 / FS)
    pred = np.column_stack([np.interp(grid, t, log[c]) for c in ("com_ap", "com_ml", "com_vert")])
    tc = com.time.to_numpy() - com.time.iloc[0]
    cg = np.arange(0, tc[-1], 1 / FS)
    true = np.column_stack([np.interp(cg, tc, com[f"CoM_Pos_{i}"].to_numpy(), left=np.nan, right=np.nan)
                            for i in (1, 2, 3)])
    if args.offset_s is None:
        coarse = best_lag(pred[:, 0], true[:, 0], int(args.max_lag_s * FS))
        n = len(grid)
        idx = np.arange(n) + coarse
        ok = (idx >= 0) & (idx < len(cg))
        shifted = np.full(n, np.nan)
        shifted[ok] = true[idx[ok], 2]
        lag = coarse + best_lag(pred[:, 2], shifted, int(0.5 * FS))
    else:
        lag = int(round(args.offset_s * FS))
    idx = np.arange(len(grid)) + lag
    ok = (idx >= 0) & (idx < len(cg))
    y = np.full_like(pred, np.nan)
    y[ok] = true[idx[ok]]
    mask = np.isfinite(y).all(1)
    m = axis_metrics(pred, np.nan_to_num(y), mask)
    exc = excursion_errors(pred, y, mask, int(args.stride_s * FS))
    for k, ax in enumerate(AXES):
        m[f"exc_rmse_{ax}_mm"] = float(1000 * np.sqrt(np.nanmean(exc[:, k] ** 2)))
    out = {"offset_s": lag / FS, "frames": int(mask.sum()), **{k: round(v, 2) for k, v in m.items()}}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
