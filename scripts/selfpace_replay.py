"""Open-loop self-paced treadmill replay: score the speed commands the lab controller would issue with each state
estimator (lab KF, trained networks) against the same controller fed with mocap CoM. Development subjects only.

Writes results/selfpace/{replay_steps.csv, replay.md, example.png}. See grf2com/selfpace.py.
Example: scripts/selfpace_replay.py --nets tcn/full_s0 gru/full_s0
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from grf2com import plotting  # noqa: E402
from grf2com import selfpace as sp  # noqa: E402
from grf2com import transitions as tr  # noqa: E402
from grf2com.config import add_config_args, config_from_args, project_path  # noqa: E402
from grf2com.data import load_trials  # noqa: E402
from grf2com.reporting import md_table  # noqa: E402
from grf2com.tables import development_subjects, final_test_subjects, load_trial_index  # noqa: E402


def load_net_preds(runs_dir: Path, spec: str) -> dict:
    model, tag = spec.split("/")
    out = {}
    for path in sorted((runs_dir / model / tag).glob("*/preds.npz")):
        with np.load(path) as z:
            for key in z.files:
                name, field = key.split("__")
                if field == "pred":
                    out[name] = z[key]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser, model_config=False)
    parser.add_argument("--nets", nargs="*", default=["tcn/full_s0"], help="model/tag runs with preds.npz")
    parser.add_argument("--example", default="AE04_T1", help="trial for the example plot")
    args = parser.parse_args()
    cfg = config_from_args(args)
    index = load_trial_index(cfg)
    dev = development_subjects(cfg, index.subject)
    assert not set(dev) & set(final_test_subjects(cfg))
    runs_dir = project_path(cfg, "runs_dir")
    nets = {spec.replace("/", "@"): load_net_preds(runs_dir, spec) for spec in args.nets}
    rows = []
    for s, t in zip(index.subject, index.trial):
        if s not in dev:
            continue
        trial = load_trials(cfg, [(s, t)])[0]
        preds = {name: p[trial.name] for name, p in nets.items() if trial.name in p}
        df = sp.replay_trial(cfg, trial, preds)
        if len(df):
            ev, _ = tr.trial_events(cfg, s, t)
            in_pulse = np.zeros(len(df), bool)
            for e in ev.itertuples():
                in_pulse |= (df.t >= e.onset).to_numpy() & (df.t < e.settle + tr.PULSE_POST_S).to_numpy()
            df["phase"] = np.where(in_pulse, "pulse", "other")
        rows.append(df)
        print(f"{s} {t}: {len(df)} steps", flush=True)
    steps = pd.concat(rows, ignore_index=True)
    estimators = ["kf"] + list(nets)
    use, offset = sp.score(steps, estimators)
    out = project_path(cfg, "results_dir") / "selfpace"
    out.mkdir(parents=True, exist_ok=True)
    use.to_csv(out / "replay_steps.csv", index=False)
    overall = sp.summarize(use, estimators)
    by_phase = sp.summarize(use, estimators, "phase")
    per_subj = sp.summarize(use, estimators, "subject")
    cols = ["estimator", "steps", "cmd_rmse_mps", "cmd_bias_mps", "cmd_corr", "pos_term_rmse_mps",
            "vel_term_rmse_mps", "com_rmse_mm", "dcom_rmse_mps", "direction_agree", "roughness_mps",
            "oracle_roughness_mps"]
    subj = per_subj.pivot(index="subject", columns="estimator", values="cmd_rmse_mps")
    md = ["# Self-paced treadmill, open-loop replay (development subjects)", "",
          "The lab controller (gain_p 0.35, gain_v 0.42, p_off -0.05 m, delta 0.3 s) acts once per detected step. Each "
          "estimator supplies com_step / dcom_step; scores are against the same controller fed with mocap CoM (the "
          "oracle). Open loop: the recorded subject never reacts to these commands, so this scores command quality, not "
          f"closed-loop behaviour. Steps in the first {sp.WARMUP_S:.0f} s after the filter starts are not scored. Frame "
          f"offset mocap → controller frame: {offset * 1000:+.0f} mm.", "",
          "- cmd: commanded target speed error vs the oracle (m/s); corr: correlation of the commanded speed changes.",
          "- pos/vel term: the part of the command error from the position / speed estimate.",
          "- roughness: RMS step-to-step change of the command (the oracle's own roughness is shown for scale).", "",
          "## All steps", md_table(overall[cols], index=False, digits=3), "",
          "## Steps during speed pulses vs the rest",
          md_table(by_phase[["phase"] + cols], index=False, digits=3), "",
          "## Command RMSE per subject (m/s)", md_table(subj, digits=3)]
    (out / "replay.md").write_text("\n".join(md), encoding="utf-8")
    print(overall[cols].round(3).to_string(index=False))

    import matplotlib.pyplot as plt
    plotting.apply_style()
    s, t = args.example.split("_")
    ex = use[(use.subject == s) & (use.trial == t) & (use.t > 240) & (use.t < 300)]
    if len(ex):
        tm = pd.read_csv(project_path(cfg, "data_dir") / f"{s}_{t}_Treadmill.csv")
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].step(tm.time, tm.speed, where="post", color=plotting.TEXT_2, lw=1, label="recorded belt")
        for c, e in enumerate(["oracle"] + estimators):
            axes[0].plot(ex.t, ex[f"{e}_v"], color=plotting.SERIES[c], lw=1.2, marker="o", ms=2,
                         label=e.replace("@", " "))
            axes[1].plot(ex.t, 1000 * ex[f"{e}_com"], color=plotting.SERIES[c], lw=1.2, label=e.replace("@", " "))
        axes[0].set_xlim(240, 300)
        axes[0].set_ylabel("target speed (m/s)")
        axes[0].set_title(f"{s} {t}: commands the lab controller would issue (open loop)")
        axes[0].legend(ncol=5, loc="lower left")
        axes[1].set_ylabel("com_step (mm, controller frame)")
        axes[1].set_xlabel("time (s)")
        plotting.save(fig, out / "example.png")


if __name__ == "__main__":
    main()
