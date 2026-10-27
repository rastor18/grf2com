"""Raw C3D force input (preprocess.grf_source: raw) and the causal prefilter, in batch and streaming."""
import copy

import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_config, write_c3d
from grf2com import preprocess as pp
from grf2com.c3d import read_raw_grf
from grf2com.realtime import StreamingPreprocessor, stream_trial
from test_realtime import offsets, raw_trial

PRE10 = "preprocess.prefilter={order: 2, cutoff_hz: 10, passes: 2}"
RAW = ["preprocess.grf_source=raw", "realtime.upstream_filter=null"]


def test_c3d_reader_round_trip(synth_root):
    csv = pd.read_csv(synth_root / "AE_grf2com_csv" / "S01_T1_GRF.csv")
    raw = read_raw_grf(synth_root / "grf_raw" / "S01_T01.c3d")
    assert len(raw) == len(csv)   # the NaN padding is dropped
    for c in raw.columns:
        np.testing.assert_allclose(raw[c], csv[c], rtol=1e-6, atol=1e-3, err_msg=c)


def test_raw_cache_equals_csv_cache_for_unfiltered_csv(synth_root):
    # the synthetic CSV is unfiltered, so both sources hold the same forces (to float32 precision)
    a = pp.load_trial(pp.run(synthetic_config(synth_root), verbose=False), "S02", "T1")
    b = pp.load_trial(pp.run(synthetic_config(synth_root, overrides=RAW), verbose=False), "S02", "T1")
    np.testing.assert_allclose(a["x"], b["x"], rtol=1e-4, atol=2e-2, equal_nan=True)
    np.testing.assert_array_equal(a["mask"], b["mask"])


def test_cache_key_unchanged_by_default_keys(synth_root):
    cfg = synthetic_config(synth_root)
    old = copy.deepcopy(cfg)
    for key in ("grf_source", "prefilter"):
        old["preprocess"].pop(key)
    assert pp.cache_key(cfg) == pp.cache_key(old)
    assert pp.cache_dir(synthetic_config(synth_root, overrides=RAW)) != pp.cache_dir(cfg)
    assert pp.cache_dir(synthetic_config(synth_root, overrides=[*RAW, PRE10])) != pp.cache_dir(
        synthetic_config(synth_root, overrides=RAW))


def test_raw_source_needs_recomputed_cop(synth_root):
    cfg = synthetic_config(synth_root, overrides=[*RAW, "preprocess.cop_from_moments=false"])
    with pytest.raises(ValueError, match="cop_from_moments"):
        pp.load_grf("S01", "T1", cfg["preprocess"], pp.project_path(cfg, "data_dir"),
                    pp.project_path(cfg, "raw_grf_dir"))


def test_prefilter_is_causal(synth_root, tmp_path):
    cfg = synthetic_config(synth_root, overrides=[*RAW, PRE10])
    csv = pd.read_csv(synth_root / "AE_grf2com_csv" / "S01_T1_GRF.csv")
    fm = {c: csv[c].to_numpy() for c in csv.columns if c[:2] in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")}
    cut = 20_000
    late = {c: np.where(np.arange(len(v)) > cut, v + 500.0, v) for c, v in fm.items()}
    out = {}
    for name, d in (("a", fm), ("b", late)):
        raw_dir = tmp_path / name
        raw_dir.mkdir()
        write_c3d(raw_dir / "S01_T01.c3d", d)
        out[name] = pp.load_grf("S01", "T1", cfg["preprocess"], pp.project_path(cfg, "data_dir"), raw_dir)
    for c in fm:
        np.testing.assert_array_equal(out["a"][c][:cut + 1], out["b"][c][:cut + 1])
        assert not np.allclose(out["a"][c][cut + 1:], out["b"][c][cut + 1:])
        if np.ptp(fm[c]) > 1:   # (the synthetic Mz is constant)
            assert not np.allclose(out["a"][c], fm[c])   # the prefilter is applied


def test_streaming_matches_batch_with_raw_and_prefilter(synth_root):
    cfg = synthetic_config(synth_root, overrides=[*RAW, PRE10])
    pp.run(cfg, verbose=False)
    t, raw, speed = raw_trial(synth_root)
    pre = StreamingPreprocessor(cfg["preprocess"], offsets(cfg))
    assert len(pre.pre) == 2
    frames, _ = stream_trial(pre, t, raw, speed)
    batch = pp.load_trial(pp.cache_dir(cfg), "S01", "T1")["x"]
    np.testing.assert_allclose(frames[:len(batch)], batch, rtol=1e-4, atol=2e-2, equal_nan=True)


def test_streaming_rejects_upstream_emulation_for_raw_models(synth_root):
    cfg = synthetic_config(synth_root, overrides=RAW)
    with pytest.raises(ValueError, match="upstream"):
        StreamingPreprocessor(cfg["preprocess"], {}, upstream_filter={"order": 2, "cutoff_hz": 10, "passes_live": 2})
