"""Per-variable marginal goodness-of-fit tests.

Two-sample Kolmogorov-Smirnov (Massey, 1951) for distributional shape
and Mann-Whitney U (Mann & Whitney, 1947) for stochastic dominance.
Bonferroni correction (Bonferroni, 1936) is applied across columns when
``adjust='bonferroni'``.

References
----------
Bonferroni, C. E. (1936). Teoria statistica delle classi e calcolo delle
    probabilita. *Pub. R. Ist. Sup. Sci. Econ. Comm. Firenze*, 8, 3-62.
Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two
    random variables is stochastically larger than the other. *Annals
    Math. Stat.*, 18(1), 50-60.
    https://doi.org/10.1214/aoms/1177730491
Massey, F. J. (1951). The Kolmogorov-Smirnov test for goodness of fit.
    *J. Amer. Stat. Assoc.*, 46(253), 68-78.
    https://doi.org/10.1080/01621459.1951.10500769
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu

Adjust = Literal["none", "bonferroni"]


def _adjust(p_values: np.ndarray, adjust: Adjust) -> np.ndarray:
    if adjust == "bonferroni":
        return cast(np.ndarray, np.clip(p_values * len(p_values), 0.0, 1.0))
    return p_values


def ks_per_variable(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    adjust: Adjust = "none",
) -> pd.DataFrame:
    """Two-sample Kolmogorov-Smirnov test per shared numeric column.

    Parameters
    ----------
    real, synthetic : pandas.DataFrame
    columns : list[str], optional
        Restrict the comparison to these columns.
    adjust : {'none', 'bonferroni'}, default 'none'

    Returns
    -------
    pandas.DataFrame
        Columns: ``feature``, ``D``, ``p_raw``, ``p_adj``.
    """
    cols = columns or sorted(set(real.columns) & set(synthetic.columns))
    rows: list[dict[str, float | str]] = []
    for c in cols:
        if not (
            pd.api.types.is_numeric_dtype(real[c]) and pd.api.types.is_numeric_dtype(synthetic[c])
        ):
            continue
        a = real[c].dropna().to_numpy()
        b = synthetic[c].dropna().to_numpy()
        if len(a) == 0 or len(b) == 0:
            continue
        stat, p = ks_2samp(a, b)
        rows.append(
            {"feature": c, "D": round(float(stat), 6), "p_raw": float(p), "p_adj": float(p)}
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["p_adj"] = _adjust(df["p_raw"].to_numpy(), adjust)
    return df


def mwu_per_variable(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    adjust: Adjust = "none",
) -> pd.DataFrame:
    """Mann-Whitney U two-sided test per shared numeric column."""
    cols = columns or sorted(set(real.columns) & set(synthetic.columns))
    rows: list[dict[str, float | str]] = []
    for c in cols:
        if not (
            pd.api.types.is_numeric_dtype(real[c]) and pd.api.types.is_numeric_dtype(synthetic[c])
        ):
            continue
        a = real[c].dropna().to_numpy()
        b = synthetic[c].dropna().to_numpy()
        if len(a) == 0 or len(b) == 0:
            continue
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        rows.append(
            {"feature": c, "U": round(float(stat), 4), "p_raw": float(p), "p_adj": float(p)}
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["p_adj"] = _adjust(df["p_raw"].to_numpy(), adjust)
    return df


__all__ = ["Adjust", "ks_per_variable", "mwu_per_variable"]
