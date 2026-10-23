"""Raw tables (infotable, trial index, gait events) and the development / locked final-test subject lists."""
from __future__ import annotations

import pandas as pd

from .config import project_path

# Never a model input (user instruction). There is no override.
FORBIDDEN_SUBJECT_FEATURES = frozenset({"dephys"})
SUBJECT_FEATURES = ("ages", "sexes", "heights", "shoes", "feet", "legs", "mass")


def check_subject_features(names) -> list[str]:
    names = list(names)
    bad = sorted(set(names) & FORBIDDEN_SUBJECT_FEATURES)
    if bad:
        raise ValueError(f"subject feature(s) {bad} must never be used as model inputs")
    unknown = sorted(set(names) - set(SUBJECT_FEATURES))
    if unknown:
        raise ValueError(f"unknown subject feature(s) {unknown}; choose from {list(SUBJECT_FEATURES)}")
    return names


def load_infotable(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(project_path(cfg, "infotable")).set_index("subjects")


def load_trial_index(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(project_path(cfg, "data_dir") / "trial_index.csv")


def load_events(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(project_path(cfg, "data_dir") / "grf_events.csv")


def load_trial_map(cfg: dict) -> pd.DataFrame | None:
    path = project_path(cfg, "trial_map")
    return pd.read_csv(path) if path.exists() else None


def final_test_subjects(cfg: dict) -> list[str]:
    return list(cfg["data"].get("final_test_subjects") or [])


def development_subjects(cfg: dict, subjects) -> list[str]:
    """All subjects except the locked final-test set, sorted."""
    locked = set(final_test_subjects(cfg))
    return sorted(s for s in set(subjects) if s not in locked)
