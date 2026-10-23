# grf2com

Predict absolute lab-frame center-of-mass (CoM) position from force-plate data, treadmill belt speed and
anthropometrics, and compare neural-network architectures (MLP, TCN, GRU, LSTM, Transformer) against classical
baselines (mean, ridge, Kalman filter, and the lab's implementation of the Song et al. 2020 treadmill state estimator,
`song_kf`). Every model and preprocessing step is causal, so a trained model can run in
real time from force plates.

## Setup (Windows)
Two virtual environments share the same code: `.venv` for the Intel Arc GPU / CPU, and `.venv-cuda` for an NVIDIA
GPU (here an RTX 4070 SUPER eGPU, 5–40× faster for training; see `results/benchmark.md`). A PyTorch install is either
XPU or CUDA, never both, hence two venvs.
```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/xpu
.venv\Scripts\python.exe -m pip install -r requirements.txt

python -m venv .venv-cuda
.venv-cuda\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv-cuda\Scripts\python.exe -m pip install -r requirements.txt
```
The code picks the device automatically (cuda → xpu → mps → cpu), or pass `--device`.

## Data
- `AE_grf2com_csv/`: per trial `*_GRF.csv` (1200 Hz force plates), `*_CoM.csv` (60 Hz mocap CoM) and
  `*_Treadmill.csv` (belt-speed command log), plus `trial_index.csv` and `grf_events.csv`. See its `README.md`.
- `infotable.csv`: one row per participant (see `infotable_README.md`). The `dephys` column is never used.
- Raw data is read-only. Everything generated goes to `cache/`, `runs/` and `results/`.

## Workflow
```powershell
.venv\Scripts\python.exe scripts\preprocess.py        # zero plates, recompute COP, causal filter, align to 60 Hz, cache
.venv\Scripts\python.exe scripts\data_report.py       # check offsets, lag, units, belt speed (development subjects)
.venv\Scripts\python.exe scripts\train.py --config configs\models\tcn.yaml --folds AE03 --dev   # quick check
.venv\Scripts\python.exe scripts\run_baselines.py     # mean, ridge, Kalman on every development fold
.venv-cuda\Scripts\python.exe scripts\sweep.py --ablations full source_cop --baselines   # resumable
.venv\Scripts\python.exe scripts\leaderboard.py       # results\leaderboard.md
.venv\Scripts\python.exe scripts\benchmark.py         # training speed per device, results\benchmark.md
.venv\Scripts\python.exe -m pytest                    # causality, data handling, end-to-end on synthetic data
.venv\Scripts\python.exe scripts\final_test.py --configs configs\models\tcn.yaml --confirm   # once, at the very end
```
Override any config value with `--set key.sub=value`, for example `--set data.features=[forces,cop]`.
Sweep ablations: `full`, `source_cop` (exported COP instead of recomputed), `no_subject`, `no_speed`, `forces_only`,
`forces_cop`, `vert_resid` / `vert_centered` (vertical = anthropometric height estimate + network; see
`target.vertical` in `configs/base.yaml`).

## COP
The exported COP columns carry a per-trial offset of metres and drift at low force, so preprocessing recomputes COP
from the zeroed moments and vertical force (`preprocess.cop_from_moments: true`). The `source_cop` ablation trains on
the exported COP instead, so the two can be compared. Details: `CLAUDE.md` and `results/data_report.md`.

## Deployment
See [DEPLOY.md](DEPLOY.md): export a model bundle trained on the development subjects, pull it on the lab PC and run it live in shadow mode next to the lab software, then score the session against mocap.

## Configs
- `configs/base.yaml`: paths, preprocessing, features, splits, training and evaluation defaults.
- `configs/models/*.yaml`: one file per architecture, merged on top of `base.yaml`.

## Evaluation
- **Locked final test set:** AE01, AE05 and AE08 (`data.final_test_subjects`) are kept out of all development work:
  splits, normalization, tuning, the sweep, the leaderboard and the data report. `scripts/final_test.py --confirm`
  retrains the chosen configs on all development subjects and scores them on the locked subjects once, writing
  `results/final/` and a run log. AE05 and AE08 fall outside the development body-size range, so expect higher errors.
- **Development:** leave-one-subject-out over the other 8 subjects (8 folds: 6 train, 1 validation, 1 test).

Whole test trials are streamed with 4 s windows and a 2 s hop, matching real-time use. Metrics are RMSE / MAE / bias / R² per axis in mm, plus excursion RMSE (both curves minus a one-stride moving
average), gait-cycle error profiles, parameter count and CPU inference latency.
