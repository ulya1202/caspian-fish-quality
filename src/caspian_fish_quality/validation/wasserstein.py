"""Per-variable 1-Wasserstein distance.

Earth-mover distance between two empirical distributions in 1D, normalised
by the empirical SD of ``real`` to give a unitless effect size. Theory
follows Villani (2009, *Optimal Transport: Old and New*, Springer).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def wasserstein_per_variable(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """Compute the 1-Wasserstein distance per shared numeric column.

    Parameters
    ----------
    real, synthetic : pandas.DataFrame
    columns : list[str], optional
    normalize : bool, default True
        If ``True``, divide each W1 by ``real[c].std()`` (effect size).

    Returns
    -------
    pandas.DataFrame
        Columns: ``feature``, ``W1``, ``W1_normalized`` (if ``normalize``).
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
        d = float(wasserstein_distance(a, b))
        row: dict[str, float | str] = {"feature": c, "W1": round(d, 6)}
        if normalize:
            sd = float(np.std(a))
            row["W1_normalized"] = round(d / sd, 6) if sd > 0 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = ["wasserstein_per_variable"]
