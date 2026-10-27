"""Run a model bundle in shadow mode: estimate CoM live (or from a replayed trial), log it, and optionally publish it
over UDP. It never commands the treadmill.

Live (lab PC, next to the lab software; plates unloaded for the first --tare-s seconds):
  scripts/live_shadow.py --bundle deploy/models/tcn_v1 --source bertec --height 172 --mass 76.5 --legs 86 --feet 25
Replay a recorded development trial (dry run; pace 1 = real time, 0 = as fast as possible):
  scripts/live_shadow.py --bundle deploy/models/tcn_v1 --source replay --trial AE02_T1 --pace 0 --duration 120

Logs go to deploy/logs/<time>_<source>.csv (t, frame, com_ap/ml/vert in m, compute_ms [, raw channels]).
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from grf2com import live  # noqa: E402
from grf2com import preprocess as pp  # noqa: E402
from grf2com.bundle import load_bundle  # noqa: E402
from grf2com.config import ROOT, load_config  # noqa: E402
from grf2com.realtime import RAW, StreamingEstimator, StreamingPreprocessor  # noqa: E402
from grf2com.tables import load_infotable  # noqa: E402

SUBJECT_ARGS = {"height": "heights", "mass": "mass", "legs": "legs", "feet": "feet", "age": "ages", "sex": "sexes",
                "shoe": "shoes"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--source", choices=["bertec", "replay"], required=True)
    parser.add_argument("--trial", help="replay: SUBJECT_TRIAL, e.g. AE02_T1 (development subjects only)")
    parser.add_argument("--pace", type=float, default=1.0, help="replay speed (1 = real time, 0 = as fast as possible)")
    for a, f in SUBJECT_ARGS.items():
        parser.add_argument(f"--{a}", type=float, help=f"subject {f} (cm / kg; as in infotable.csv)")
    parser.add_argument("--subject-json", help="JSON file with the infotable fields instead of the flags")
    parser.add_argument("--tare-s", type=float, default=3.0, help="bertec: seconds of unloaded plates to tare from")
    parser.add_argument("--no-tare", action="store_true", help="bertec: trust Bertec AUTOZERO (offsets 0)")
    parser.add_argument("--upstream", choices=["bundle", "off"], help="causal emulation of the training data's "
                        "upstream filter (default: bundle for bertec, off for replay of already-filtered CSVs)")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--udp", help="publish estimates as JSON to HOST:PORT")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--log", help="log CSV path")
    parser.add_argument("--log-raw", action="store_true", help="also log the raw channels (for check_live_mapping)")
    args = parser.parse_args()

    model, fb, cfg, meta = load_bundle(args.bundle)
    print(f"bundle {Path(args.bundle).name}: {meta['model']}, trained on {meta['train_subjects']}, inputs "
          f"{meta['grf_rate_hz']} Hz, needs subject fields {meta['required_subject_fields']}")
    here = load_config()
    if args.subject_json:
        subject = json.loads(Path(args.subject_json).read_text())
    elif args.source == "replay" and all(getattr(args, a) is None for a in SUBJECT_ARGS):
        subject = load_infotable(here).loc[args.trial.split("_")[0]].to_dict()
    else:
        subject = {f: getattr(args, a) for a, f in SUBJECT_ARGS.items() if getattr(args, a) is not None}
    missing = [f for f in meta["required_subject_fields"] if f not in subject]
    if missing:
        sys.exit(f"missing subject fields {missing} (flags: {', '.join('--' + a for a in SUBJECT_ARGS)})")

    if args.source == "replay":
        if not args.trial:
            sys.exit("--trial is required for replay")
        s, t = args.trial.split("_")
        source = live.ReplaySource(here, s, t, args.pace, cfg["preprocess"].get("grf_source", "csv"))
        tare = json.loads((pp.cache_dir(cfg) / f"{s}_{t}.json").read_text())
        offsets, fs = {c: tare[f"off_{c}"] for c in RAW}, source.fs
        upstream = args.upstream or "off"
    else:
        # the channel mapping comes from the current configs/base.yaml, so mapping fixes need no re-export
        source = live.BertecSource(here["realtime"]["bertec"], args.host, args.port)
        print("listening to the Bertec data stream ...", flush=True)
        if args.no_tare:
            offsets, fs = {c: 0.0 for c in RAW}, None
        else:
            print(f"taring: keep both plates unloaded for {args.tare_s:.0f} s", flush=True)
            offsets, fs = live.measure_offsets(iter(source), args.tare_s)
        print(f"message keys: {source.first_keys}", flush=True)
        if not fs or not np.isfinite(fs):
            _, fs = live.measure_offsets(iter(source), 1.0)
        print(f"input rate: {fs:.0f} Hz", flush=True)
        source.fs = fs                  # spaces queued samples (sample clock) if processing falls behind
        upstream = args.upstream or "bundle"
    up = meta.get("upstream_filter") if upstream == "bundle" else None
    pre = StreamingPreprocessor(cfg["preprocess"], offsets, fs=float(fs), upstream_filter=up)
    est = StreamingEstimator(model, fb, cfg, subject)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log) if args.log else ROOT / "deploy" / "logs" / f"{stamp}_{args.source}.csv"
    print(f"running (upstream filter emulation: {up or 'off'}); log: {log_path}", flush=True)
    try:
        summary = live.run(iter(source), pre, est, log_path, args.udp, args.duration, args.log_raw)
    except KeyboardInterrupt:
        summary = {"interrupted": True}
    if args.source == "replay" and "frames" in summary:
        cdir = pp.cache_dir(cfg)
        d = pp.load_trial(cdir, *args.trial.split("_"))
        pred = pd.read_csv(log_path)[["com_ap", "com_ml", "com_vert"]].to_numpy()
        n = min(len(pred), len(d["y"]))
        m = d["mask"][:n]
        err = 1000 * (pred[:n][m] - d["y"][:n][m])
        summary["rmse_mm_vs_mocap"] = dict(zip(["ap", "ml", "vert"], np.sqrt((err ** 2).mean(0)).round(1).tolist()))
        summary["note"] = "the bundle was trained on all development subjects, so this replay is in-sample"
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
