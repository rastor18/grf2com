"""Alignment, valid mask, belt speed, COP recompute, training-only normalization, locked subjects, dephys guard."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import NAN_FRAMES, OFF_TREADMILL_S, PLATE_X, SPEED_LOG, synthetic_config
from grf2com import preprocess as pp
from grf2com.config import parse_override
from grf2com.data import CH, FeatureBuilder, final_fold, load_trials, make_folds
from grf2com.tables import check_subject_features, load_infotable


@pytest.fixture(scope="module")
def cfg(synth_root):
    return synthetic_config(synth_root)


def test_targets_are_aligned_and_masked(cfg, synth_root):
    d = pp.load_trial(pp.cache_dir(cfg), "S01", "T1")
    com = pd.read_csv(synth_root / "AE_grf2com_csv" / "S01_T1_CoM.csv")
    n = len(d["y"])
    np.testing.assert_allclose(d["y"], com[["CoM_Pos_1", "CoM_Pos_2", "CoM_Pos_3"]].to_numpy()[:n], rtol=1e-6)
    assert not d["mask"][NAN_FRAMES].any()                                    # CoM NaN -> invalid
    assert not d["mask"][-int((OFF_TREADMILL_S - 0.5) * 60):].any()           # off the plates -> invalid
    assert d["mask"][100:500].all()


def test_belt_speed_is_a_causal_hold(cfg):
    d = pp.load_trial(pp.cache_dir(cfg), "S01", "T1")
    speed = d["x"][:, CH["speed"]]
    t = np.arange(len(speed)) / 60
    for (t0, v0), (t1, _) in zip(SPEED_LOG[:-1], SPEED_LOG[1:]):
        inside = (t >= t0) & (t < t1)
        assert np.allclose(speed[inside], v0)
    assert speed[0] == 0.0


def test_cop_recomputed_or_source(cfg, synth_root):
    # The synthetic source COPx has a +3000 mm offset; the recompute recovers the true plate position.
    x = pp.load_trial(pp.cache_dir(cfg), "S01", "T1")["x"]
    stance = x[:, CH["contact_l"]] > 0.99
    assert abs(np.median(x[stance, CH["COPx_l"]]) - PLATE_X["left"]) < 5
    src_cfg = synthetic_config(synth_root, overrides=["preprocess.cop_from_moments=false"])
    assert pp.cache_dir(src_cfg) != pp.cache_dir(cfg)
    pp.run(src_cfg, trials=[("S01", "T1")], verbose=False)
    xs = pp.load_trial(pp.cache_dir(src_cfg), "S01", "T1")["x"]
    assert abs(np.median(xs[stance, CH["COPx_l"]]) - (PLATE_X["left"] + 3000)) < 5


def test_normalization_uses_training_subjects_only(cfg):
    info = load_infotable(cfg)
    fold = make_folds(cfg)[0]
    train = load_trials(cfg, fold.train)
    fb = FeatureBuilder(cfg).fit(train, info)
    assert fb.state["train_subjects"] == fold.subjects("train")
    assert not set(fb.state["train_subjects"]) & set(fold.subjects("test") + fold.subjects("val"))
    # Changing a test trial cannot change the statistics.
    test = load_trials(cfg, fold.test)
    test[0].x[:] = 1e6
    fb2 = FeatureBuilder(cfg).fit(train, info)
    assert fb2.state["mean"] == fb.state["mean"]


@pytest.mark.parametrize("mode", ["anthro_residual", "anthro_centered"])
def test_vertical_anthro_modes(synth_root, mode):
    cfg = synthetic_config(synth_root, overrides=[f"target.vertical={mode}"])
    info = load_infotable(cfg)
    fold = make_folds(cfg)[0]
    train, test = load_trials(cfg, fold.train), load_trials(cfg, fold.test)
    fb = FeatureBuilder(cfg).fit(train, info)
    # The height model is fit on training subjects only.
    assert sorted(fb.state["height"]["subject_means"]) == fold.subjects("train")
    if mode == "anthro_centered":
        # Each training subject's vertical target is centred on that subject's own mean height.
        for s in fold.subjects("train"):
            y = np.concatenate([fb._offset_y(t)[t.mask, 2] for t in train if t.subject == s])
            assert abs(y.mean()) < 1e-6
    # Test subjects always use the anthropometric estimate, and denormalize adds it back.
    t = test[0]
    est = fb._height.predict(t.subject, info)
    assert fb.vertical_offset(t.subject, for_target=True) == pytest.approx(est)
    z = fb.target(t)
    back = fb.denormalize(z, t.subject)
    np.testing.assert_allclose(back[t.mask], t.y[t.mask], atol=1e-5)
    # Round trip through a checkpointed state.
    fb2 = FeatureBuilder(cfg).load_state(fb.state)
    np.testing.assert_allclose(fb2.denormalize(z, t.subject), back)
    with pytest.raises(ValueError, match="subject"):
        fb.denormalize(z)


def test_locked_subjects_never_in_development_folds(cfg):
    for split in ("loso", "trial_holdout"):
        c = dict(cfg, data={**cfg["data"], "split": split})
        for f in make_folds(c) + make_folds(c, dev_mode=True):
            assert "S05" not in f.subjects("train") + f.subjects("val") + f.subjects("test")
    assert [f.name for f in make_folds(cfg)] == ["S01", "S02", "S03", "S04"]
    final = final_fold(cfg)
    assert final.subjects("test") == ["S05"] and "S05" not in final.subjects("train")


def test_dephys_guard():
    with pytest.raises(ValueError, match="never"):
        check_subject_features(["heights", "dephys"])
    cfg = synthetic_config(Path("unused"), overrides=["data.subject_features=[heights,dephys]"])
    with pytest.raises(ValueError, match="never"):
        FeatureBuilder(cfg)


def test_set_override_parsing():
    assert parse_override("train.lr=3e-4") == {"train": {"lr": 3e-4}}
    assert parse_override("data.features=[forces,cop]") == {"data": {"features": ["forces", "cop"]}}
    assert parse_override("data.subject_features=[]") == {"data": {"subject_features": []}}


def test_single_belt_inputs(cfg, synth_root):
    from grf2com.data import combined_cop, is_single_belt
    single = synthetic_config(synth_root, overrides=["data.features=[total_force,single_cop,single_moments,speed]"])
    assert is_single_belt(single) and not is_single_belt(cfg)
    fb = FeatureBuilder(single)
    assert fb.names == ["Fx_tot", "Fy_tot", "Fz_tot", "COPx_c", "COPy_c", "Mx_c", "My_c", "speed"]
    x = pp.load_trial(pp.cache_dir(cfg), "S01", "T1")["x"]
    cx, cy, fz = combined_cop(x, 508.0)
    only_l = (x[:, CH["contact_l"]] > 0.99) & (x[:, CH["Fz_r"]] < 1)
    only_r = (x[:, CH["contact_r"]] > 0.99) & (x[:, CH["Fz_l"]] < 1)
    # Single support: the combined COP is that foot's COP shifted into the common frame (synthetic plates at -/+460).
    assert abs(np.median(cx[only_l]) - (PLATE_X["left"] + 508)) < 5
    assert abs(np.median(cx[only_r]) - (PLATE_X["right"] - 508)) < 5
    np.testing.assert_allclose(fz, x[:, CH["Fz_l"]] + x[:, CH["Fz_r"]])
    info = load_infotable(single)
    fold = make_folds(single)[0]
    fb.fit(load_trials(single, fold.train), info)
    assert np.isfinite(fb.transform(load_trials(single, fold.test)[0])).all()
