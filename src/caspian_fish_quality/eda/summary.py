"""Lightweight DataFrame summary helpers used during EDA."""

from __future__ import annotations

from typing import cast

import pandas as pd


def table_shapes(df_dict: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Return a DataFrame with ``rows``/``cols`` for every table.

    Parameters
    ----------
    df_dict : dict[int, pandas.DataFrame]

    Returns
    -------
    pandas.DataFrame
        Indexed by ``key`` with columns ``rows`` and ``cols``.
    """
    rows = [{"key": key, "rows": df.shape[0], "cols": df.shape[1]} for key, df in df_dict.items()]
    return cast("pd.DataFrame", pd.DataFrame(rows).set_index("key").sort_index())


def describe_dict(
    df_dict: dict[int, pd.DataFrame], *, include_object: bool = False
) -> dict[int, pd.DataFrame]:
    """Apply :meth:`pandas.DataFrame.describe` to every table.

    Parameters
    ----------
    df_dict : dict[int, pandas.DataFrame]
    include_object : bool, default False
        If ``True``, include object columns in ``describe``.

    Returns
    -------
    dict[int, pandas.DataFrame]
    """
    out: dict[int, pd.DataFrame] = {}
    for key, df in df_dict.items():
        out[key] = df.describe(include="all" if include_object else None)
    return out


__all__ = ["describe_dict", "table_shapes"]
