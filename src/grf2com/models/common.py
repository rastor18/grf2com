"""Shared model pieces: causal convolution, channel norm, FiLM, and the base class handling subject fusion.

Every model maps inputs x (B, L, C) and a subject vector s (B, S) to outputs (B, L, 3), one per input sample, and
output t may depend only on inputs <= t.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

FUSIONS = ("none", "input", "head", "film")


class CausalConv1d(nn.Conv1d):
    """Conv1d on (B, C, L) with left padding only, so output t sees inputs <= t."""

    def __init__(self, cin: int, cout: int, kernel: int, dilation: int = 1):
        super().__init__(cin, cout, kernel, dilation=dilation)
        self.left = (kernel - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.left, 0)))


class ChannelNorm(nn.Module):
    """LayerNorm over channels at each time step of (B, C, L); per time step, so causal."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.LayerNorm(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class FiLM(nn.Module):
    """Feature-wise affine modulation from the subject vector; starts as the identity."""

    def __init__(self, subj_dim: int, ch: int):
        super().__init__()
        self.lin = nn.Linear(subj_dim, 2 * ch)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, h: torch.Tensor, s: torch.Tensor, channel_dim: int = -1) -> torch.Tensor:
        gamma, beta = self.lin(s).chunk(2, dim=-1)
        if channel_dim == 1:   # h is (B, C, L)
            gamma, beta = gamma[:, :, None], beta[:, :, None]
        else:                  # h is (B, L, C)
            gamma, beta = gamma[:, None], beta[:, None]
        return h * (1 + gamma) + beta


class SeqModel(nn.Module):
    """Base class. Subclasses set self.feat_dim, build their backbone on self.backbone_in channels and implement
    features(x, s) -> (B, L, feat_dim). Subject fusion:
      none  - subject vector ignored
      input - appended to every input sample
      head  - appended to the backbone features before the output layer
      film  - FiLM on the backbone features (models with film_inside apply it inside their blocks instead)
    """

    film_inside = False

    def __init__(self, in_ch: int, subj_dim: int, fusion: str):
        super().__init__()
        if fusion not in FUSIONS:
            raise ValueError(f"subject_fusion must be one of {FUSIONS}, got {fusion!r}")
        self.fusion = fusion if subj_dim > 0 else "none"
        self.subj_dim = subj_dim
        self.backbone_in = in_ch + (subj_dim if self.fusion == "input" else 0)

    def build_head(self, feat_dim: int, out_dim: int = 3) -> None:
        self.feat_dim = feat_dim
        if self.fusion == "film" and not self.film_inside:
            self.film = FiLM(self.subj_dim, feat_dim)
        self.head = nn.Linear(feat_dim + (self.subj_dim if self.fusion == "head" else 0), out_dim)

    @property
    def block_film_dim(self) -> int:
        """Subject dimension for FiLM inside blocks (0 = no FiLM there)."""
        return self.subj_dim if (self.fusion == "film" and self.film_inside) else 0

    def features(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if self.fusion == "input":
            x = torch.cat([x, s[:, None].expand(-1, x.shape[1], -1)], dim=-1)
        h = self.features(x, s)
        if self.fusion == "film" and not self.film_inside:
            h = self.film(h, s)
        if self.fusion == "head":
            h = torch.cat([h, s[:, None].expand(-1, h.shape[1], -1)], dim=-1)
        return self.head(h)

    def receptive_field(self) -> int | None:
        """Input samples each output can see (None = unbounded, e.g. RNNs and the transformer within a window)."""
        return None
