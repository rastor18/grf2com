"""The real-time pipeline must reproduce the batch pipeline: same 60 Hz frames, same model outputs."""
import json

import numpy as np
import pandas as pd
import torch

from conftest import synthetic_config
from grf2com import preprocess as pp
from grf2com.data import FeatureBuilder, load_trials, make_folds
from grf2com.models import build_model
from grf2com.realtime import RAW, StreamingEstimator, StreamingPreprocessor, stream_trial
from grf2com.tables import load_infotable

N_PRED = 400


def raw_trial(root, subject="S01", trial="T1"):
    data = root / "AE_grf2com_csv"
    grf = pd.read_csv(data / f"{subject}_{trial}_GRF.csv")
    tm = pd.read_csv(data / f"{subject}_{trial}_Treadmill.csv")
    t = grf.time.to_numpy()
    raw = {c: grf[f"{c.split('_')[0]}_{pp.SIDES[c[-1]]}"].to_numpy() for c in RAW}
    idx = np.searchsorted(tm.time.to_numpy(), t, side="right") - 1
    speed = np.where(idx >= 0, tm.speed.to_numpy()[np.maximum(idx, 0)], 0.0)
    return t, raw, speed


def offsets(cfg, subject="S01", trial="T1"):
    info = json.loads((pp.cache_dir(cfg) / f"{subject}_{trial}.json").read_text())
    return {c: info[f"off_{c}"] for c in RAW}


def test_streaming_preprocessor_matches_batch(synth_root):
    cfg = synthetic_config(synth_root)
    t, raw, speed = raw_trial(synth_root)
    pre = StreamingPreprocessor(cfg["preprocess"], offsets(cfg))
    frames, _ = stream_trial(pre, t, raw, speed)
    batch = pp.load_trial(pp.cache_dir(cfg), "S01", "T1")["x"]
    assert len(frames) >= len(batch)
    np.testing.assert_allclose(frames[:len(batch)], batch, rtol=1e-5, atol=1e-3, equal_nan=True)


def test_streaming_estimator_matches_batch(synth_root):
    cfg = synthetic_config(synth_root, "tcn")
    info = load_infotable(cfg)
    fold = make_folds(cfg)[0]
    fb = FeatureBuilder(cfg).fit(load_trials(cfg, fold.train), info)
    torch.manual_seed(0)
    model = build_model(cfg, fb.n_inputs, fb.n_subject).eval()
    trial = load_trials(cfg, [("S01", "T1")])[0]
    subject = info.loc["S01"].to_dict()
    est = StreamingEstimator(model, fb, cfg, subject)
    live = np.array([est.push(f) for f in trial.x[:N_PRED]])

    feats = torch.from_numpy(fb.transform(trial))
    s = torch.from_numpy(fb.subject_vector("S01", info))[None]
    window = int(round(cfg["eval"]["window_s"] * pp.COM_RATE))
    ref = []
    with torch.inference_mode():
        for j in range(N_PRED):
            x = feats[max(0, j - window + 1):j + 1][None]
            ref.append(fb.denormalize(model(x, s)[0, -1].numpy()[None], est.LIVE)[0])   # same anthropometrics as S01
    np.testing.assert_allclose(live, np.array(ref), rtol=1e-5, atol=1e-6)
