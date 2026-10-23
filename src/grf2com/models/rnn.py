"""Unidirectional GRU / LSTM. Bidirectional is rejected because it is not causal."""
from __future__ import annotations

import torch
from torch import nn

from .common import SeqModel


class RNN(SeqModel):
    def __init__(self, in_ch: int, subj_dim: int, mcfg: dict, rate_mult: int, cell: str = "gru"):
        super().__init__(in_ch, subj_dim, mcfg.get("subject_fusion", "input"))
        if mcfg.get("bidirectional", False):
            raise ValueError("bidirectional RNNs are not causal")
        hidden, layers = int(mcfg["hidden"]), int(mcfg["layers"])
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.rnn = rnn_cls(self.backbone_in, hidden, layers, batch_first=True,
                           dropout=float(mcfg.get("dropout", 0.0)) if layers > 1 else 0.0)
        self.build_head(hidden)

    def features(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return self.rnn(x)[0]
