"""Merge tables 1, 2, 3, 6 into one wide ``static`` DataFrame.

Each fish becomes one row carrying biometric (``bio_*``), index (``idx_*``),
cut/yield (``cut_*``), and fatty-acid (``fa_*``) features side by side.
"""

from __future__ import annotations

from functools import reduce
from typing import cast

import pandas as pd

_PREFIXES: dict[int, str] = {1: "bio", 2: "idx", 3: "cut", 6: "fa"}


def merge_static(synthetic_dict: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate tables 1, 2, 3, 6 by group and per-group sample id.

    Parameters
    ----------
    synthetic_dict : dict[int, pandas.DataFrame]
        Output of :func:`caspian_fish_quality.synth.generators.generate_all_synthetic`.

    Returns
    -------
    pandas.DataFrame
        One row per virtual fish; ``group`` is the leading column.
    """
    frames: list[pd.DataFrame] = []
    for key in (1, 2, 3, 6):
        if key not in synthetic_dict:
            continue
        df = synthetic_dict[key].copy()
        df["sample_id"] = df.groupby("group").cumcount()
        prefix = _PREFIXES[key]
        rename = {c: f"{prefix}_{c}" for c in df.columns if c not in ("group", "sample_id")}
        frames.append(df.rename(columns=rename))

    if not frames:
        return pd.DataFrame()

    merged = reduce(
        lambda left, right: left.merge(right, on=["group", "sample_id"], how="outer"),
        frames,
    )
    merged = merged.drop(columns=["sample_id"])
    cols = ["group", *[c for c in merged.columns if c != "group"]]
    return cast("pd.DataFrame", merged[cols])


__all__ = ["merge_static"]
