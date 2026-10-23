"""Transformer encoder with a causal attention mask and sinusoidal positions over the window."""
from __future__ import annotations

import math

import torch
from torch import nn

from .common import SeqModel


def sinusoidal(length: int, dim: int, device) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: dim // 2])
    return pe


class Transformer(SeqModel):
    def __init__(self, in_ch: int, subj_dim: int, mcfg: dict, rate_mult: int):
        super().__init__(in_ch, subj_dim, mcfg.get("subject_fusion", "input"))
        d = int(mcfg["d_model"])
        self.inp = nn.Linear(self.backbone_in, d)
        layer = nn.TransformerEncoderLayer(d, int(mcfg["nhead"]), int(mcfg["dim_feedforward"]),
                                           float(mcfg.get("dropout", 0.0)), activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, int(mcfg["layers"]), enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.build_head(d)

    def features(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        h = self.inp(x) + sinusoidal(length, self.inp.out_features, x.device)
        mask = nn.Transformer.generate_square_subsequent_mask(length, device=x.device)
        return self.norm(self.encoder(h, mask=mask, is_causal=True))
