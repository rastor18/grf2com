# Deploying a model in the lab (shadow mode)

The loop: **train / retrain here → export a bundle → commit and push → `git pull` on the lab PC → run it live in
shadow mode → score the session against mocap → repeat.** Shadow mode only estimates and logs the CoM (optionally
publishing it over UDP). It never commands the treadmill, and it reads the Bertec data stream as a second subscriber
next to the lab software.

## 1. Export a model bundle (here)
```powershell
.venv-cuda\Scripts\python.exe scripts\export_model.py --config configs\models\tcn.yaml --tag full_s0 --name tcn_v1
```
- Trains on every **development** trial (the locked final-test subjects are never loaded), for the median best epoch
  of that config's LOSO runs (`--epochs` to override). It takes about a minute on the eGPU.
- Writes `deploy/models/tcn_v1/model.pt` + `bundle.json` (~0.5 MB): inputs, rates, required subject fields, the
  upstream filter the training data had, the development LOSO scores and the code revision.
- Bundles are tracked in git (private repo), so redeploying is commit → push → `git pull` on the lab PC.

## 2. Lab PC setup (once)
```powershell
git clone <private repo url> grf2com
cd grf2com
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt pyzmq
```
Inference runs on the CPU: one prediction takes ~3–8 ms and preprocessing ~1–3 % of a core. The 4090 is better used
for training (a second venv with the CUDA wheel, like `.venv-cuda` here).

Checks on the lab PC:
- `.venv\Scripts\python.exe -m pytest` (the realtime equivalence tests use the synthetic dataset; data-dependent tests
  skip themselves without the raw data).
- With a copy of the data folder: a replay dry run,
  `.venv\Scripts\python.exe scripts\live_shadow.py --bundle deploy\models\tcn_v1 --source replay --trial AE02_T1 --pace 1 --duration 60`

## 3. Verify the live channel mapping (once per setup, and whenever the export pipeline changes)
The models were trained on the QTM → MATLAB export. The Bertec stream may use other signs or frames. In
`configs/base.yaml` → `realtime.bertec.channels`, each of our channels maps to a Bertec message key and a sign; this
mapping is **not verified yet**. It is read from the current config at run time, so a fix needs no re-export.
1. Record one trial both ways: QTM recording as usual, plus
   `scripts\live_shadow.py --bundle ... --source bertec --log-raw --height .. --mass .. --legs .. --feet ..`
2. Export that QTM recording to a `*_GRF.csv` like the training data (without the MATLAB filter is fine; it only
   compares signs and scales).
3. `scripts\check_live_mapping.py --live deploy\logs\<log>.csv --export <session>_GRF.csv`. This flags channels
   whose sign or scale disagrees; fix them in `realtime.bertec.channels` and re-check.

## 4. Run live (shadow mode)
```powershell
.venv\Scripts\python.exe scripts\live_shadow.py --bundle deploy\models\tcn_v1 --source bertec `
    --height 172 --mass 76.5 --legs 86 --feet 25 --udp 127.0.0.1:9870
```
- The first `--tare-s` (3) seconds tare the plates: **keep both plates unloaded**. Use `--no-tare` to trust Bertec
  AUTOZERO instead.
- It prints the Bertec message keys and the measured input rate; the filters are designed for the measured rate.
- `--upstream bundle` (the default for live) emulates the training data's MATLAB filter causally (see below).
- Subject fields are in the infotable units (cm, kg). The bundle lists which ones it needs.
- Logs: `deploy/logs/<time>_bertec.csv` with `t, frame, com_ap, com_ml, com_vert` (m, mocap frame) and compute time;
  `--log-raw` adds the raw channels. Ctrl+C stops it. `--udp HOST:PORT` publishes `{"t", "frame", "com"}` JSON per
  frame for other programs.
- Don't use the locked participants (AE01, AE05, AE08) for live tests.

## 5. Score the session
Export the session's mocap CoM like the training data (`*_CoM.csv`), then
`scripts\score_live_log.py --log deploy\logs\<log>.csv --com <session>_CoM.csv`. The clocks are aligned
automatically (AP drift, then vertical oscillation), or with `--offset-s`.

## Known gaps (read before trusting live numbers)
- **Training data were filtered zero-phase** (MATLAB `filtfilt`, 2nd order, 10 Hz), so the inputs used future
  samples. Live, only a causal filter is possible: about 45 ms lag at the matching smoothness
  (`realtime.upstream_filter.passes_live: 2`). Expect live accuracy below the offline numbers until the models are
  retrained on unfiltered data. With unfiltered training data, set `realtime.upstream_filter: null` and re-export.
- **Channel mapping**, the Bertec publish rate and second-subscriber access are unverified on the lab hardware
  (step 3).
- The streaming runtime doesn't support the `lab_kf` / `kalman_state` feature groups yet.
- Shadow mode only. Feeding the estimate into the treadmill controller needs a separate safety review: fallback to
  the KF, plausibility checks, speed limits.

## Retrain → redeploy
Change configs or data, re-run the LOSO sweep if you want comparable scores (`scripts/sweep.py`), export with a new
`--name`, commit and push, then `git pull` on the lab PC. Each bundle records its training subjects, inputs and code
revision in `bundle.json`, so logs can be traced back to the model that produced them.
