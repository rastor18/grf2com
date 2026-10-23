"""Model registry. build_model(cfg, in_ch, subj_dim) returns a SeqModel: (B, L, C) + (B, S) -> (B, L, 3)."""
from __future__ import annotations

from functools import partial

from .common import FUSIONS, SeqModel
from .mlp import MLP
from .rnn import RNN
from .tcn import TCN
from .transformer import Transformer

MODELS = {
    "mlp": MLP,
    "tcn": TCN,
    "gru": partial(RNN, cell="gru"),
    "lstm": partial(RNN, cell="lstm"),
    "transformer": Transformer,
}


def build_model(cfg: dict, in_ch: int, subj_dim: int) -> SeqModel:
    mcfg = cfg["model"]
    name = mcfg["name"]
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; available: {sorted(MODELS)}")
    rate_mult = int(cfg["preprocess"]["grf_rate"]) // 60
    return MODELS[name](in_ch, subj_dim, mcfg, rate_mult)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["FUSIONS", "MODELS", "SeqModel", "build_model", "count_parameters"]
