"""Past-window MLP: a causal Conv1d over the last context_s seconds, then pointwise layers."""
from __future__ import annotations

import torch
from torch import nn

from .common import CausalConv1d, SeqModel


class MLP(SeqModel):
    def __init__(self, in_ch: int, subj_dim: int, mcfg: dict, rate_mult: int):
        super().__init__(in_ch, subj_dim, mcfg.get("subject_fusion", "input"))
        hidden, layers, p = int(mcfg["hidden"]), int(mcfg["layers"]), float(mcfg.get("dropout", 0.0))
        self.context = max(1, int(round(float(mcfg["context_s"]) * 60 * rate_mult)))
        mods: list[nn.Module] = [CausalConv1d(self.backbone_in, hidden, self.context)]
        for _ in range(layers - 1):
            mods += [nn.GELU(), nn.Dropout(p), nn.Conv1d(hidden, hidden, 1)]
        mods += [nn.GELU(), nn.Dropout(p)]
        self.net = nn.Sequential(*mods)
        self.build_head(hidden)

    def features(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).transpose(1, 2)

    def receptive_field(self) -> int:
        return self.context
