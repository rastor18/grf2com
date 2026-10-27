"""Check the raw C3Ds against the CSV export: the CSV forces and moments should be filtfilt(Butterworth 2nd order, 10 Hz)
of the raw channels plus one constant per channel (the two exports zero the plates differently; our tare removes it).

Per trial: row counts, per-channel constant, residual RMS / max after removing it (1 s trimmed at each end for the
filter edge transients), and the raw noise above 10 Hz (RMS of raw - filtfilt(raw)). Also lists the C3Ds that have no
CSV trial (T00 trials, other studies, re-recordings). Writes results/raw_grf_check.{csv,md}. Raw data is only read.
Locked subjects are included: this is a data-integrity check, no statistics or modelling.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grf2com import preprocess as pp  # noqa: E402
from grf2com.c3d import c3d_name, read_raw_grf  # noqa: E402
from grf2com.config import add_config_args, config_from_args, project_path  # noqa: E402
from grf2com.reporting import md_table  # noqa: E402

CHANNELS = [f"{q}_{side}" for side in pp.SIDES.values() for q in pp.TARED]
TRIM = pp.GRF_RATE   # 1 s


def check_one(subject: str, trial: str, data_dir: Path, raw_dir: Path, n_csv: int, file: str | None = None) -> dict:
    path = raw_dir / (file or c3d_name(subject, trial))
    row = {"subject": subject, "trial": trial, "file": path.name, "n_csv": n_csv}
    try:
        raw = read_raw_grf(path)
    except Exception as e:   # noqa: BLE001 - reported, not raised
        return {**row, "error": f"{type(e).__name__}: {e}"}
    row["n_raw"] = len(raw)
    if len(raw) < n_csv:
        return {**row, "error": "raw shorter than the CSV"}
    csv = pd.read_csv(data_dir / f"{subject}_{trial}_GRF.csv", engine="pyarrow")
    sos = signal.butter(2, 10, fs=pp.GRF_RATE, output="sos")
    b, a = signal.butter(2, 10, fs=pp.GRF_RATE)
    worst_rms, worst_max = 0.0, 0.0
    for c in CHANNELS:
        x = raw[c].to_numpy(float)[:n_csv]
        smooth = signal.filtfilt(b, a, x)
        err = (smooth - csv[c].to_numpy(float))[TRIM:-TRIM]
        const = float(np.median(err))
        err = err - const
        row[f"const_{c}"] = const
        row[f"rms_{c}"] = float(np.sqrt(np.mean(err ** 2)))
        worst_rms, worst_max = max(worst_rms, row[f"rms_{c}"]), max(worst_max, float(np.abs(err).max()))
        row[f"noise_{c}"] = float(np.std(x - signal.sosfiltfilt(sos, x)))
    row.update({"worst_rms": worst_rms, "worst_max": worst_max, "error": ""})
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser, model_config=False)
    parser.add_argument("--tol", type=float, default=0.05, help="pass if every channel's residual RMS is below (N, N*m)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cfg = config_from_args(args)
    from joblib import Parallel, delayed

    data_dir, raw_dir = project_path(cfg, "data_dir"), project_path(cfg, "raw_grf_dir")
    index = pd.read_csv(data_dir / "trial_index.csv")
    jobs = [(s, t, int(n), None) for s, t, n in zip(index.subject, index.trial, index.n_grf_samples)]
    expected = {c3d_name(s, t) for s, t, _, _ in jobs}
    extras = sorted(p.name for p in raw_dir.glob("*.c3d") if p.name not in expected)
    # re-recordings of a CSV trial (e.g. AE07_T03_v1, AE09_T06A): check them too, to see which file the CSV came from
    for name in extras:
        stem = name[:-4]
        for s, t, n, _ in list(jobs):
            base = c3d_name(s, t)[:-4]
            if stem.startswith(base) and stem != base:
                jobs.append((s, t, n, name))
    rows = Parallel(n_jobs=args.workers)(delayed(check_one)(s, t, data_dir, raw_dir, n, f) for s, t, n, f in jobs)
    df = pd.DataFrame(rows)
    df["pass"] = (df["error"] == "") & (df["worst_rms"] < args.tol)
    results = project_path(cfg, "results_dir")
    results.mkdir(parents=True, exist_ok=True)
    df.to_csv(results / "raw_grf_check.csv", index=False)

    main_rows = df[df.file == [c3d_name(s, t) for s, t in zip(df.subject, df.trial)]]
    noise = {c: df[f"noise_{c}"].median() for c in CHANNELS if f"noise_{c}" in df}
    const = {c: df[f"const_{c}"].median() for c in CHANNELS if f"const_{c}" in df}
    show = df[["subject", "trial", "file", "n_csv", "n_raw", "worst_rms", "worst_max", "pass", "error"]]
    md = ["# Raw C3D vs CSV GRF check", "",
          f"CSV = filtfilt(Butterworth 2nd order, 10 Hz) of the raw channel + one constant per channel; pass = every "
          f"channel's residual RMS < {args.tol} (N or N*m), 1 s trimmed at each end.", "",
          f"- CSV trials with a passing C3D: {int(main_rows['pass'].sum())} / {len(index)}",
          f"- Failing or unreadable: {', '.join(main_rows.loc[~main_rows['pass'], 'file']) or 'none'}",
          f"- C3Ds without a CSV trial (ignored): {', '.join(extras) or 'none'}", "",
          "## Median per channel over trials", "",
          md_table(pd.DataFrame({"raw noise above 10 Hz (RMS)": noise, "constant CSV - raw": const}), 2), "",
          "## Per file", "", md_table(show, 4, index=False), ""]
    (results / "raw_grf_check.md").write_text("\n".join(md))
    print("\n".join(md[:8]))
    sys.exit(0 if main_rows["pass"].all() and len(main_rows) == len(index) else 1)


if __name__ == "__main__":
    main()
