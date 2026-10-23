# grf2com

Experiment environment for comparing neural-network architectures that predict **absolute lab-frame CoM position**
(`CoM_Pos_1..3`) from force-plate signals (GRF / COP / moments), treadmill belt speed and anthropometrics.
The target use is **real-time with force plates**, so every model and every preprocessing step must be **causal**
(an output may depend only on current and past samples).

## Permissions (from the user)
- Full access to create, modify or delete anything inside this folder without asking. In auto mode, proceed without
  permission prompts for in-folder work.
- **Raw data is read-only:** the CSVs in `AE_grf2com_csv/` and `infotable.csv`. Never write to them.

## Status
- Code is complete: preprocessing, data report, data module, 5 models, training / streaming evaluation, baselines,
  sweep, leaderboard, benchmark, final test, tests (`pytest`: 54 pass, on a synthetic dataset).
- Two venvs: `.venv` (torch 2.14.1+xpu: CPU + Intel Arc) and `.venv-cuda` (torch 2.14.1+cu130: CPU + NVIDIA RTX
  4070 SUPER eGPU). Same code; use `.venv-cuda` when the eGPU is connected (5–40× faster, `results/benchmark.md`).
- Full development LOSO sweep done (5 models × 8 folds × {recomputed COP `full_s0`, source COP `source_cop_s0`},
  plus baselines `full` / `source_cop`; 68 min on the RTX 4070 SUPER; log `results/sweep_log.txt`).
- `.venv-cuda` Python has exited with code 127 *after* finishing long CUDA runs (sweep, benchmark) although all
  output was written; not reproducible with short runs. Don't chain further commands after it with `&&`.
- Check the eGPU with a standalone `nvidia-smi -L` (allowed without a prompt in `.claude/settings.json`).

## Results so far (development LOSO, 8 folds; `results/leaderboard.md`)
- Best: TCN 17.8 ± 8.0 mm mean RMSE (AP 14.3, ML 7.1, vertical 32.0), excursion ~5 mm, 3.6 ms CPU per window.
  Transformer 18.9, LSTM / MLP 20.1, GRU 21.2, ridge 21.8, Kalman 25.4, mean 61.3.
- **COP:** recomputed beats source COP for every model on average, but only slightly for the networks (TCN 17.8 vs
  18.0; AP 14.3 vs 15.6, recomputed better in 6/8 folds), because the tared moments are also inputs and carry the
  same information (COP = M/Fz). The Kalman baseline, which uses COP alone for position, collapses with source COP
  (AP 41 → 122 mm, equal to the mean baseline). So `source_cop_s0` is not a "source COP only" test.
- **Song et al. (2020) baseline (`song_kf`, `src/grf2com/baselines/song_kf.py`)** replays the lab's implementation,
  `C:\Users\rish5\Documents\self-paced-treadmill-VT_visual_feedback\src\kalman_filter.py`. The user said: where it
  differs from the paper, follow the lab code. That means: 30 % BW contact threshold, a 15 ms loop with A/B/Q built for
  0.01 s, a 3.6 s step gate, start at treadmill start, mass calibrated from standing, and no standing guard. It is AP
  only (ML / vertical come from the mean baseline), with one COP→CoM offset fit on training subjects.
  Development LOSO AP 85.8 ± 42.6 mm (mean baseline 125.6, our Kalman 40.7, TCN 14.3). With source COP it is unusable
  (AP 1.3 m). Paper-style step-mean metrics: median 0.036 m / 0.058 m/s (paper 0.014 m / 0.023 m/s). It matches the
  paper in AE04 / AE11 (≈0.017 m, 0.03 m/s) but not in AE02 / AE03, where the foot-midpoint measurement barely follows
  the CoM. Lab code quirks worth raising with the user: the prediction runs at 2/3 of real time (model Δt 0.01 vs a
  15 ms timer), the gate was "temporarily" widened to 3.6 s, and there is no guard when contacts stop (AE06 T6 exit).
  The load-dependent Fy bias (≈ −3 N while walking, not removed by the swing-phase tare) biases the speed estimate.
- **Speed-transition scoring** (`scripts/transition_report.py`, `src/grf2com/transitions.py`; evaluation only):
  - Pulses: every development trial has ±0.5 m/s pulses (1.25 → 0.75/1.75 → 1.25, edges ~0.43 s, return after 3 s,
    next pulse ~7 s later) from ~255 s on: 1502 pulses in total.
  - Errors rise during the pulse and are back to baseline within ~1 s. So the headline window is onset → settle + 1 s;
    the 2 s / 8 s window dilutes the effect.
  - Default runs, pulse AP RMSE: TCN 17.0, transformer 16.8, GRU 16.8, MLP 18.0, LSTM 22.0 mm (ratio to steady
    1.19–1.28).
  - The discriminating metric is pulse AP *shape*: TCN 7.7, transformer 8.9, GRU 9.0, MLP 9.9, LSTM 11.5. The TCN is
    better than GRU, MLP and LSTM in 8/8 folds (transformer 5/8).
  - Within-stride AP motion is only ~6 mm RMS. A constant (mean baseline) scores 6.4 steady / 8.7 pulse, so most
    networks barely beat a constant on AP shape. The Kalman (force integration) is best at 5.7 / 7.0, which hints that
    a hybrid with the networks would help.
  - Steady comes from earlier in the trial than the pulses (time-in-trial isn't controlled). All pulses are the same
    size, so there's no breakdown by size.
  - Not done yet: oversampling transitions in training (plan step 5).
- **Self-paced treadmill platform, open-loop replay** (`scripts/selfpace_replay.py`, `src/grf2com/selfpace.py`;
  `results/selfpace/replay.md`). The lab's controller law, at lab defaults (gain_p 0.35, gain_v 0.42, p_off −0.05 m,
  delta 0.3 s, v 0–2.6 m/s, a ≤ 1.5 m/s², from `reset_speed_controller_param`), acts on every KF-detected step. It is
  fed by three estimators: the lab KF, network predictions (out of fold), or mocap CoM (the "oracle").
  - The KF port matches the lab's `kalman_filter.py` tick for tick (`tests/test_lab_equivalence.py`, which imports
    the lab code with stubbed Qt). The first tick is one 15 ms timer period after init, as in the lab.
  - Results over 53k steps: command error vs the oracle is KF 0.049 m/s (corr 0.66) vs TCN 0.011 (0.97), transformer
    0.012, GRU 0.013. During pulses: KF 0.067 vs TCN 0.014.
  - The KF matches the networks in AE04 / AE11 (0.016) and fails in AE02 / AE03 (cross-stepping subjects).
  - Caveats: it's open loop (the subject never reacts to the commands). The networks are trained on the oracle's own
    target, whereas the KF tracks the foot midpoint by design.
  - Mocap → controller frame offset: one constant, the median of (KF com_step − mocap step mean).
  - Closed loop (option B, a stride-library walker) was proposed but not built; the user chose open loop only.
- **TCN + lab KF state** (`plus_labkf_s0` vs `full_s0`): the input adds the lab KF's p, v and time since the last
  step, run over the whole trial.
  - Overall 15.1 → 14.8 mm (better in 4/8 folds; noise). AP 15.7 → 15.1, mixed (AE04 −9.6, AE03 +4.9).
  - AP movement shape improves consistently: steady 6.2 → 5.6 and pulses 7.7 → 7.5, better in 7/8 folds. The one
    exception is AE03, the cross-stepping subject where the KF is unreliable.
  - The `kalman_state` group (our Kalman) is implemented but not run, by user choice.
- **Single-belt result** (`single_belt_s0` vs `full_s0`, paired per fold):
  - Losing the split belt costs little overall: TCN 15.1 → 15.9 mm (worse in 4/8 folds), transformer 15.5 → 16.3,
    MLP 15.5 → 15.3, GRU 15.9 → 18.1, LSTM 17.3 → 20.3.
  - The consistent loss is ML: +1.7 to +4.1 mm, worse in 5–7/8 folds. Not knowing which foot is down costs lateral
    information. AP and vertical are about unchanged.
  - GRU loses movement shape in 8/8 folds; TCN, transformer and MLP barely do. The Kalman is worse with combined COP
    (AP 40.7 → 46.9).
  - Pulse AP shape: TCN 7.7 → 7.9, transformer 8.9 → 8.5, MLP 9.9 → 9.5, GRU 9.0 → 11.0.
- **Run tags after the default change:** `full_s0` = default (`anthro_centered`); `vert_absolute_s0` and
  `source_cop_vabs_s0` = the earlier absolute-vertical runs.
- **Vertical offset experiment (`vert_resid_s0`, centered = now the default `full_s0`):**
  - `anthro_residual` doesn't help (TCN vertical 31.7 vs 32.0 mm) and hurts AP.
  - `anthro_centered` brings every network's vertical error down to 23–25 mm, and overall error to 15.1–17.3 mm
    (TCN 15.1, from 17.8). AP is about unchanged and vertical movement shape slightly better (TCN 7/8 folds).
  - The gain is uneven across folds (better in only 3–6/8): the vertical offset now just equals the height regression's
    error (AE06 −68 mm, AE03 +40 mm), which forces can't fix.
  - The floor is about 22 mm (the Kalman's 21.7 uses the same regression). Going lower needs a per-subject reference:
    a calibration value, like Song's p0.
- **Vertical is the largest error and is almost all a per-subject constant offset** (87–95% of the networks'
  vertical MSE). Networks (32–41 mm) are worse than the anthropometric height regression alone (mean baseline
  27.3 mm; Kalman 21.7 mm). Example: fold AE07 offset +73 to +92 mm for networks vs +4 mm for the regression.

## Data facts
- 86 trials (AE01–AE11 × T1–T8; AE05 T8 and AE10 T4 are absent), ~8 min each. Index: `AE_grf2com_csv/trial_index.csv`.
- `*_GRF.csv`: 1200 Hz, `time` + `F{x,y,z}`, `COP{x,y,z}`, `M{x,y,z}` for `_left` / `_right` plates.
- `*_CoM.csv`: 60 Hz mocap, `frame` (1-based), `time`, `CoM_Pos_1..3`. Missing values are written as `NaN`.
- **Alignment:** CoM frame j (0-based) ↔ GRF row 20·j. Force and mocap look offset by about 1 frame (17 ms).
- **Plates are not zeroed.** Swing-phase offsets are Fx ≈ 68/146 N, Fy ≈ 295/291 N, Fz ≈ 21 N, My(right) ≈ 733.
  Zero each plate per trial (median of each F/M channel while that plate is unloaded).
- **Axes (inferred from correlations):** Fx ↔ CoM_Pos_2 (ML), Fy ↔ CoM_Pos_1 (AP), Fz ↔ CoM_Pos_3 (vertical).
- **Units:** CoM in m. COP in mm. **Moments are N·m** (source COP ≈ 1000·M/Fz). COPz is all zero, so drop it.
- **Source COP columns are unusable as absolute position.** They carry a trial-specific constant offset of metres
  (per-trial median COPy 0.7–8.8 m, left-right gaps up to ±6 m) and drift at low force; during swing they hold a
  steady plateau, never 0. Preprocessing recomputes COP from tared channels by default
  (`COPx = -1000·My/Fz`, `COPy = 1000·Mx/Fz`, `preprocess.cop_from_moments: true`; `false` = source COP, run as the
  `source_cop` ablation because the user wants the deployable option compared on data). Recomputed COP is consistent:
  plates at x ≈ ∓465 mm (sd ~20 mm across trials); COPy correlates 0.99 with CoM AP across trials (~120 mm ahead).
  Live, forces and moments stream too, so recomputed COP is available in deployment.
- **Upstream filter (from the user):** the dataset was filtered in MATLAB at 1200 Hz with `filtfilt` and a
  2nd-order Butterworth at 10 Hz. That filter is zero-phase, so every input sample already contains about 50–100 ms
  of future information, and offline results are optimistic for real time. A causal equivalent lags about 22 ms (one
  pass) or about 45 ms (two passes, matching the `filtfilt` magnitude). If COP was filtered as its own channel after
  QTM computed it, that explains the source vs recomputed COP difference at low force. (Asked the user; not yet
  confirmed.)
- **Training vs deployment gap:** the CSV forces and moments were filtered upstream (zero-phase, see above), so
  training inputs are not strictly causal and are smoother than a raw live stream. Faithful real-time training needs an
  unfiltered export (only our causal filter applied). The tare is a whole-trial median (≈ a pre-trial zero, drift
  ≤ 1.5 N); `preprocess.tare_window_s` restricts it to the first N seconds.
- **Source vs recomputed COP (identification on development trials):** pipeline is Bertec → QTM → Visual3D → CSV,
  F/M filtered (spectra are 130–146 dB down by 100 Hz). Mid-stance (Fz > 200 N): source = recomputed + one constant
  per plate per trial (−0.2 to +7.9 m; within-trial spread 5–14 mm over all 63 trials; unrelated to the zero offsets).
  Below ~200 N the source drifts 50–200 mm toward a stable swing plateau. No static formula (free origin, scale, depth,
  M and Fz offsets), time shift, Butterworth, or free-form ±100 ms FIR of the recomputed COP reproduces that drift
  (residual 55–170 mm for COPy), so it is nonlinear: most likely COP = M/Fz taken on differently processed
  (unfiltered and/or differently zeroed) signals, with a low-force plateau, then filtered. Settling it needs the
  unfiltered QTM export for one trial.
- **Data report findings (development subjects, `results/data_report.md`):** plate drift ≤ 1.5 N over a trial;
  mean total Fz is 1.04–1.12 × infotable body weight (~4–9 kg carried, likely the exo); cached force lags mocap CoM
  acceleration by ~24 ms vertical / ~20 ms ML (~9 ms of that is the causal filter); stride-averaged recomputed COP
  predicts stride-averaged CoM across subjects at 29 mm AP / 8 mm ML RMSE (mean baseline: 121 / 25 mm).
- **Valid targets:** CoM not NaN AND total zeroed Fz > 0.3·BW. AE06 T6 ends with the subject off the treadmill
  (CoM NaN for the last ~4.4k frames, both Fz at offset level).
- **Belt speed:** `*_Treadmill.csv` (`time, speed, frame, label`) is a step function of speed commands on the
  force-plate timeline, 0 to 1.75 m/s, ~160–840 rows per trial. There is **one** speed column (no per-belt speed);
  `label` equals `speed` in all 86 trials. Resample with causal hold:
  `speed[np.searchsorted(tm.time, t, side="right") - 1]`, and 0 before the first command.
  `*_TmSpeeds.csv` (per-stride resampling) is **not used**.
- `grf_events.csv`: `HS_/TO_{left,right}`, `sample_index` = row into that trial's GRF file (values 2 to ~602k,
  probably 1-based). Events are for evaluation and baselines only, **never** model inputs.
- `infotable.csv`: `subjects, ages, sexes, heights (cm), shoes, dephys, feet (cm), legs (cm), mass (kg)`.
  **Never use `dephys` as a model input** (hard guard, no override).
- **Locked final test set: AE01, AE05, AE08** (`data.final_test_subjects`). Never use them in development: no splits,
  normalization or subject-vector statistics, baseline tuning, sweeps, leaderboard or data report. Only
  `scripts/final_test.py --confirm` loads them, once, at the very end, and only when the user asks. AE05 (52.6 kg) and
  AE08 (180 cm, 96 cm legs) are outside the development body-size range. AE01 T1 was used in the early data checks
  before it was locked.
- T1–T8 are four exo controllers (OFF, PBC, VNMC, MLE) in reverse-counterbalanced blocks (`trial_map_README.md`).
  `trial_map.csv` isn't in this folder yet. Once it is, the controller becomes an evaluation breakdown, and trials
  flagged `skip` are reviewed before use.

## Conventions
- No hard-coded paths: everything resolves relative to the project root (`configs/base.yaml` → `paths`).
- Causal only: `lfilter` not `filtfilt`, forward-fill (hold) not interpolation, causal convs and masks, no
  bidirectional RNNs.
- Normalize with **training-subject statistics only**, never per trial (absolute COP carries position information).
- Splits: leave-one-subject-out (LOSO) over the 8 development subjects by default (6 train, 1 validation, 1 test);
  validation = next development subject in rotation.
- Generated outputs go to `cache/`, `runs/<model>/<tag>/<fold>/` and `results/`. Raw data is never written.
- Config = `configs/base.yaml` deep-merged with a model file from `configs/models/`, plus `--set key.sub=value` overrides.

## Code map
- `src/grf2com/`: `config.py` (merge + `--set`), `device.py`, `tables.py` (infotable, trial index, locked/dev
  subjects, dephys guard), `preprocess.py`, `data.py` (FeatureBuilder, folds, on-device window sampler),
  `models/` (registry: mlp, tcn, gru, lstm, transformer; subject fusion none/input/head/film), `train.py`,
  `evaluate.py` (streaming eval, metrics, gait profiles, latency, plots), `baselines/` (mean, ridge, kalman,
  song_kf),
  `final.py` (locked-subject test), `anthro.py` (height regression on training subjects), `plotting.py`,
  `reporting.py`.
- `target.vertical`: `absolute` (default) | `anthro_residual` | `anthro_centered`. The anthro modes train the network
  on vertical CoM minus an offset (the anthropometric estimate, or each training subject's own mean height) and add
  the anthropometric estimate back at prediction, so scores stay in absolute height.
- Runs: `runs/<model>/<tag>/<fold>/` with `config.json`, `history.csv`, `best.pt`, `metrics.json`, `per_trial.csv`,
  `preds.npz`, `curves.png`, `trace.png`, `gait_profile.png`. Sweep tags: `<ablation>_s<seed>`; baselines `<ablation>`.
- Sweep ablations (`scripts/sweep.py`): `full`, `source_cop`, `no_subject`, `no_speed`, `forces_only`, `forces_cop`,
  `vert_resid`, `vert_absolute`, `source_cop_vabs`, `single_belt`.
- **Single-belt inputs** (`single_belt` ablation): simulates a regular, non-split instrumented treadmill. Inputs are
  the summed forces (`total_force`), the combined COP (`single_cop`) and the moments about a common origin
  (`single_moments`: Mx = Fz·COPy, My = −Fz·COPx), plus speed. There are no per-plate channels and no left/right
  contact flags, and no combined Mz.
  - Combined COP = the Fz-weighted mean of per-plate COP after shifting ML by ±`data.belt_ml_offset_mm` (508 mm, the
    lab code's `belt_width`). Checked: step width 87 ± 33 mm, and the foot midpoint correlates 0.97 with CoM ML.
  - Ridge and Kalman use the combined inputs; `song_kf` is skipped (`split_belt_only`).

## Deployment (see DEPLOY.md)
- Loop: `scripts/export_model.py` (train on all development trials → `deploy/models/<name>/`, ~0.5 MB, tracked in git)
  → push → `git pull` on the lab PC → `scripts/live_shadow.py --source bertec` (shadow mode, never commands the
  treadmill) → `scripts/score_live_log.py` against the session's mocap CoM.
- `src/grf2com/realtime.py`: `StreamingPreprocessor` buffers raw samples and processes one block per frame, exactly
  equal to the batch cache (`tests/test_realtime.py`; 12–25 µs per sample). `StreamingEstimator` runs the model on
  the last 4 s every frame (~7 ms median CPU on the laptop). `src/grf2com/live.py`: replay / Bertec sources, tare,
  run loop. `src/grf2com/bundle.py`: bundle write/load.
- Live data: Bertec ZMQ publisher (`bertec_remote_control.py`, port 5556), subscribed without conflation as a
  second, read-only subscriber. The lab's own loop keeps only the latest sample every 15 ms. The publish rate is
  measured at start-up.
- `realtime.bertec.channels` (our channel → Bertec key, sign) is unverified; check it with
  `scripts/check_live_mapping.py`. `realtime.upstream_filter` describes the training data's MATLAB `filtfilt`; live,
  it is emulated causally (2 passes, ~45 ms lag). Set it to null once models are trained on unfiltered data.
- Dry runs: in-sample replay of AE02 T1 at 8.2 / 2.4 / 7.1 mm; it keeps up in real time with UDP publishing on.
- Repo: `git init` done, nothing committed. The repo is private and the data are deidentified (user), so docs can
  mention participants. Raw data, cache/, runs/, results/ and deploy/logs/ stay out of git.

## Commands
Use a venv interpreter: `.venv/Scripts/python.exe` (Intel / CPU) or `.venv-cuda/Scripts/python.exe` (NVIDIA eGPU).
```
.venv/Scripts/python.exe scripts/preprocess.py                 # build cache/<hash>/ from the raw CSVs (~40 s)
.venv/Scripts/python.exe scripts/data_report.py                # results/data_report.{csv,md} + plots
.venv/Scripts/python.exe scripts/train.py --config configs/models/tcn.yaml --folds AE03 --dev
.venv/Scripts/python.exe scripts/run_baselines.py              # mean / ridge / Kalman, full LOSO
.venv-cuda/Scripts/python.exe scripts/sweep.py --ablations full source_cop --baselines   # resumable
.venv/Scripts/python.exe scripts/transition_report.py          # results/transitions.{csv,md}, speed-pulse scoring
.venv/Scripts/python.exe scripts/selfpace_replay.py --nets tcn/full_s0   # results/selfpace/, controller command replay
.venv/Scripts/python.exe scripts/leaderboard.py                # results/leaderboard.{csv,md} (+ transitions table)
.venv/Scripts/python.exe scripts/benchmark.py                  # merges into results/benchmark.{csv,md}
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe scripts/final_test.py --configs ... --confirm   # locked subjects; only when the user asks
.venv-cuda/Scripts/python.exe scripts/export_model.py --config configs/models/tcn.yaml --tag full_s0 --name tcn_v1
.venv/Scripts/python.exe scripts/live_shadow.py --bundle deploy/models/tcn_v1 --source replay --trial AE02_T1 --pace 0
```

## Machine
Core Ultra 7 256V (8 cores), 16 GB RAM, Intel Arc 140V iGPU (PyTorch XPU), and an NVIDIA GeForce RTX 4070 SUPER
(12 GB) eGPU when connected (`.venv-cuda`). Device selection order: cuda → xpu → mps → cpu (`train.device: auto`).
