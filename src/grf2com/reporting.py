"""Small helpers for Markdown reports (no extra dependencies)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _fmt(v, digits: int) -> str:
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else f"{v:.{digits}f}"
    return str(v)


def md_table(df: pd.DataFrame, digits: int = 3, index: bool = True) -> str:
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v, digits) for v in row) + " |")
    return "\n".join(lines)
