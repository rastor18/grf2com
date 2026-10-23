"""Classical baselines. Each has fit(train, val, info) and predict(trial, info) -> (n, 3) CoM in m, all causal."""
from .kalman import KalmanBaseline
from .mean import MeanBaseline
from .ridge import RidgeBaseline
from .song_kf import SongBaseline

BASELINES = {"mean": MeanBaseline, "ridge": RidgeBaseline, "kalman": KalmanBaseline, "song_kf": SongBaseline}

__all__ = ["BASELINES", "KalmanBaseline", "MeanBaseline", "RidgeBaseline", "SongBaseline"]
