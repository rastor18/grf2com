"""Perturbing future inputs must leave past outputs unchanged, for every model and every causal preprocessing step."""
import numpy as np
import pytest
import torch
from scipy import signal

from grf2com.config import ROOT, load_config
from grf2com.models import FUSIONS, MODELS, build_model
from grf2com.preprocess import causal_filter, hold_last

C, S, L, T0 = 7, 3, 64, 40


def _model(name, fusion, rate=60):
    cfg = load_config(ROOT / "configs" / "models" / f"{name}.yaml",
                      [f"model.subject_fusion={fusion}", f"preprocess.grf_rate={rate}", "model.dropout=0.0"])
    torch.manual_seed(0)
    return build_model(cfg, C, S).eval()


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("fusion", FUSIONS)
@pytest.mark.parametrize("rate", [60, 120])
def test_future_inputs_do_not_change_past_outputs(name, fusion, rate):
    model = _model(name, fusion, rate)
    x = torch.randn(2, L, C)
    s = torch.randn(2, S)
    x2 = x.clone()
    x2[:, T0 + 1:] += 10 * torch.randn_like(x2[:, T0 + 1:])
    with torch.no_grad():
        a, b = model(x, s), model(x2, s)
    assert a.shape == (2, L, 3)
    torch.testing.assert_close(a[:, :T0 + 1], b[:, :T0 + 1], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(a[:, T0 + 1:], b[:, T0 + 1:])   # the perturbation does reach later outputs


def test_bidirectional_rnn_rejected():
    cfg = load_config(ROOT / "configs" / "models" / "gru.yaml", ["model.bidirectional=true"])
    with pytest.raises(ValueError, match="causal"):
        build_model(cfg, C, S)


def test_preprocessing_filters_are_causal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    valid = rng.random(500) > 0.3
    x2 = x.copy()
    x2[301:] += 100
    sos = signal.butter(2, 25, fs=1200, output="sos")
    np.testing.assert_allclose(causal_filter(x, sos)[:301], causal_filter(x2, sos)[:301])
    np.testing.assert_allclose(hold_last(x, valid)[:301], hold_last(x2, valid)[:301])
