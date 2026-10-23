"""Ridge regression on causal lagged features (same inputs and normalization as the networks).

Each frame uses the features at lags of 0..64 frames (current and past only) plus the subject vector. The
regularization strength is chosen on the validation subject.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import FeatureBuilder, rate_mult

LAGS = [0, 1, 2, 4, 8, 16, 32, 64]
ALPHAS = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4]


class RidgeBaseline:
    name = "ridge"

    def __init__(self, cfg: dict, alpha: float | None = None):
        """alpha: fixed regularization instead of choosing it on the validation subject (used by the final test)."""
        self.cfg = cfg
        self.k = rate_mult(cfg)
        self.fixed_alpha = alpha

    def _design(self, trial, info: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Design matrix (n, F) and a flag for frames whose full lag history exists."""
        x = self.fb.transform(trial)[::self.k].astype(np.float64)
        n = len(x)
        cols = []
        for lag in LAGS:
            shifted = np.zeros_like(x)
            shifted[lag:] = x[:n - lag] if lag else x
            cols.append(shifted)
        s = self.fb.subject_vector(trial.subject, info).astype(np.float64)
        cols.append(np.tile(s, (n, 1)))
        cols.append(np.ones((n, 1)))
        full_history = np.arange(n) >= max(LAGS)
        return np.concatenate(cols, axis=1), full_history

    def fit(self, train, val, info: pd.DataFrame) -> "RidgeBaseline":
        self.fb = FeatureBuilder(self.cfg).fit(train, info)
        xtx, xty = None, None
        for t in train:
            X, ok = self._design(t, info)
            use = ok & t.mask
            Xu, yu = X[use], t.y[use].astype(np.float64)
            xtx = Xu.T @ Xu if xtx is None else xtx + Xu.T @ Xu
            xty = Xu.T @ yu if xty is None else xty + Xu.T @ yu
        n_feat = xtx.shape[0]
        penalty = np.eye(n_feat)
        penalty[-1, -1] = 0.0   # intercept is not penalized
        scale = np.trace(xtx) / n_feat
        best = None
        for a in [self.fixed_alpha] if self.fixed_alpha is not None else ALPHAS:
            w = np.linalg.solve(xtx + a * scale * 1e-3 * penalty, xty)
            if val and self.fixed_alpha is None:
                err = np.concatenate([(self._design(t, info)[0] @ w - t.y)[t.mask] for t in val])
                rmse = float(np.sqrt((err ** 2).mean(0)).mean())
            else:
                rmse = 0.0
            if best is None or rmse < best[0]:
                best = (rmse, a, w)
        self.val_rmse, self.alpha, self.w = best
        self.n_params = int(self.w.size)
        return self

    def predict(self, trial, info: pd.DataFrame) -> np.ndarray:
        return self._design(trial, info)[0] @ self.w
