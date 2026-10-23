"""Constant prediction: training mean for AP / ML, and vertical CoM height regressed on anthropometrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..anthro import HEIGHT_FEATURES, SubjectHeight


class MeanBaseline:
    name = "mean"

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def fit(self, train, val, info: pd.DataFrame) -> "MeanBaseline":
        self.mean = np.concatenate([t.y[t.mask] for t in train]).mean(0)
        self.height = SubjectHeight().fit(train, info)
        self.n_params = 2 + 1 + len(HEIGHT_FEATURES)
        return self

    def predict(self, trial, info: pd.DataFrame) -> np.ndarray:
        out = np.tile(self.mean, (trial.n_frames, 1))
        out[:, 2] = self.height.predict(trial.subject, info)
        return out
