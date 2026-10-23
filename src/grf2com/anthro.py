"""Anthropometric estimate of a subject's mean vertical CoM height (used by baselines and the vertical target option)."""
from __future__ import annotations

import numpy as np
import pandas as pd

HEIGHT_FEATURES = ["heights", "legs"]


class SubjectHeight:
    """Ridge regression of a subject's mean vertical CoM on standardized height and leg length, fit on the subjects
    of the trials passed to fit() (training subjects only)."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, trials, info: pd.DataFrame) -> "SubjectHeight":
        subjects = sorted({t.subject for t in trials})
        y = np.array([np.mean(np.concatenate([t.y[t.mask, 2] for t in trials if t.subject == s])) for s in subjects])
        X = info.loc[subjects, HEIGHT_FEATURES].to_numpy(float)
        self.mu, self.sd = X.mean(0), np.where(X.std(0) > 0, X.std(0), 1.0)
        Z = (X - self.mu) / self.sd
        self.b0 = float(y.mean())
        self.w = np.linalg.solve(Z.T @ Z + self.alpha * np.eye(Z.shape[1]), Z.T @ (y - self.b0))
        self.subject_means = dict(zip(subjects, y.tolist()))
        return self

    def predict(self, subject: str, info: pd.DataFrame) -> float:
        z = (info.loc[subject, HEIGHT_FEATURES].to_numpy(float) - self.mu) / self.sd
        return float(self.b0 + z @ self.w)

    def state(self) -> dict:
        return {"alpha": self.alpha, "mu": self.mu.tolist(), "sd": self.sd.tolist(), "b0": self.b0,
                "w": self.w.tolist(), "subject_means": self.subject_means}

    @classmethod
    def from_state(cls, state: dict) -> "SubjectHeight":
        h = cls(state["alpha"])
        h.mu, h.sd, h.w = np.asarray(state["mu"]), np.asarray(state["sd"]), np.asarray(state["w"])
        h.b0, h.subject_means = state["b0"], dict(state["subject_means"])
        return h
