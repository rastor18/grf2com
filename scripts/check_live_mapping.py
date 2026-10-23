"""Check the Bertec live channel mapping (realtime.bertec.channels) against the training-format export.

Record one session both ways: run live_shadow.py --source bertec --log-raw while QTM records, then export that QTM
recording to a *_GRF.csv in the same format as the training data. This script aligns the two by total Fz and, for each
channel, reports the correlation and the scale of live vs export, suggesting a sign flip or a unit problem.

Example: scripts/check_live_mapping.py --live deploy/logs/20261024_101500_bertec.csv --export path/to/AE12_T1_GRF.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import signal  # noqa: E402

from grf2com import preprocess as pp  # noqa: E402
from grf2com.realtime import RAW  # noqa: E402

FS = 100.0


def resample(t: np.ndarray, x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, t, x)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", required=True, help="live_shadow.py log written with --log-raw")
    parser.add_argument("--export", required=True, help="the same session exported like *_GRF.csv")
    parser.add_argument("--max-lag-s", type=float, default=30.0)
    args = parser.parse_args()
    lv = pd.read_csv(args.live)
    if not set(RAW) <= set(lv.columns):
        sys.exit("the live log has no raw channels; record it with --log-raw")
    ex = pd.read_csv(args.export, engine="pyarrow")
    col = {c: f"{c.split('_')[0]}_{pp.SIDES[c[-1]]}" for c in RAW}
    tl, te = lv.t.to_numpy() - lv.t.iloc[0], ex.time.to_numpy() - ex.time.iloc[0]
    gl, ge = np.arange(0, tl[-1], 1 / FS), np.arange(0, te[-1], 1 / FS)
    fz_l = resample(tl, (lv.Fz_l + lv.Fz_r).to_numpy(), gl)
    fz_e = resample(te, (ex[col["Fz_l"]] + ex[col["Fz_r"]]).to_numpy(), ge)
    a, b = fz_l - fz_l.mean(), fz_e - fz_e.mean()
    xc = signal.correlate(b, a, mode="full")
    lags = signal.correlation_lags(len(b), len(a), mode="full")
    ok = np.abs(lags) <= args.max_lag_s * FS
    lag = lags[ok][np.argmax(xc[ok])]
    print(f"alignment: export is {lag / FS:+.2f} s relative to the live log")
    rows = []
    for c in RAW:
        lx = resample(tl, lv[c].to_numpy(), gl)
        ix = np.arange(len(gl)) + lag
        keep = (ix >= 0) & (ix < len(ge))
        ey = resample(te, ex[col[c]].to_numpy(), ge)[ix[keep]]
        lx = lx[keep]
        r = np.corrcoef(lx - lx.mean(), ey - ey.mean())[0, 1]
        slope = np.polyfit(lx, ey, 1)[0] if lx.std() > 0 else np.nan
        verdict = "ok" if r > 0.9 and 0.8 < slope < 1.25 else "FLIP SIGN" if r < -0.9 else "CHECK"
        rows.append({"channel": c, "corr": round(r, 3), "export_per_live": round(slope, 3), "verdict": verdict})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    if (out.verdict != "ok").any():
        print("\nFix realtime.bertec.channels in configs/base.yaml (flip the sign or the key), then re-check.")


if __name__ == "__main__":
    main()
