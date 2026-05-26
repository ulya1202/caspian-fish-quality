"""Pivot tables 4, 5 into one wide ``storage`` DataFrame.

Each ``(measure, day)`` pair becomes a column. Tables produce columns
prefixed with ``stor_`` (Table 4) and ``chem_`` (Table 5).
"""

from __future__ import annotations

from typing import cast

import pandas as pd

_PREFIXES: dict[int, str] = {4: "stor", 5: "chem"}


def merge_storage(synthetic_dict: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Pivot tables 4 and 5 wide by storage day.

    Parameters
    ----------
    synthetic_dict : dict[int, pandas.DataFrame]

    Returns
    -------
    pandas.DataFrame
        One row per virtual fish (group, sample_id) with one column per
        ``(measure, day)`` combination; ``group`` is the leading column.
    """
    parts: list[pd.DataFrame] = []
    for key in (4, 5):
        if key not in synthetic_dict:
            continue
        df = synthetic_dict[key].copy()
        day_col = "storage_day" if "storage_day" in df.columns else "storage"
        measures = [c for c in df.columns if c not in ("group", day_col)]
        df["sample_id"] = df.groupby([day_col, "group"]).cumcount()
        prefix = _PREFIXES[key]

        for mc in measures:
            piv = df.pivot_table(
                index=["group", "sample_id"],
                columns=day_col,
                values=mc,
                aggfunc="first",
            )
            piv.columns = [f"{prefix}_{mc}_day{int(d)}" for d in piv.columns]
            parts.append(piv)

    if not parts:
        return pd.DataFrame()

    wide = pd.concat(parts, axis=1).reset_index()
    wide = wide.sort_values(["group", "sample_id"]).reset_index(drop=True)
    cols = ["group", "sample_id", *[c for c in wide.columns if c not in ("group", "sample_id")]]
    return cast("pd.DataFrame", wide[cols])


__all__ = ["merge_storage"]
