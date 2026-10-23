"""Dilated causal residual TCN. Dilations are multiplied by the input rate / 60 so the receptive field stays the same
length in seconds. With subject_fusion: film, every block is modulated by the subject vector."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .common import CausalConv1d, ChannelNorm, FiLM, SeqModel


class TCNBlock(nn.Module):
    def __init__(self, ch: int, kernel: int, dilation: int, dropout: float, film_dim: int):
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, kernel, dilation)
        self.norm1 = ChannelNorm(ch)
        self.conv2 = CausalConv1d(ch, ch, kernel, dilation)
        self.norm2 = ChannelNorm(ch)
        self.drop = nn.Dropout(dropout)
        self.film = FiLM(film_dim, ch) if film_dim else None

    def forward(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        y = self.drop(F.gelu(self.norm1(self.conv1(h))))
        y = self.norm2(self.conv2(y))
        if self.film is not None:
            y = self.film(y, s, channel_dim=1)
        return F.gelu(h + self.drop(y))


class TCN(SeqModel):
    film_inside = True

    def __init__(self, in_ch: int, subj_dim: int, mcfg: dict, rate_mult: int):
        super().__init__(in_ch, subj_dim, mcfg.get("subject_fusion", "film"))
        ch, kernel, p = int(mcfg["channels"]), int(mcfg["kernel_size"]), float(mcfg.get("dropout", 0.0))
        self.dilations = [int(d) * rate_mult for d in mcfg["dilations"]]
        self.kernel = kernel
        self.inp = nn.Conv1d(self.backbone_in, ch, 1)
        self.blocks = nn.ModuleList(TCNBlock(ch, kernel, d, p, self.block_film_dim) for d in self.dilations)
        self.build_head(ch)

    def features(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h = self.inp(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h, s)
        return h.transpose(1, 2)

    def receptive_field(self) -> int:
        return 1 + 2 * (self.kernel - 1) * sum(self.dilations)
