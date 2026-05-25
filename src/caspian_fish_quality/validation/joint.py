"""Joint-distribution validation via correlation-matrix distances.

Frobenius norm of the difference between target and empirical Pearson
correlation matrices, plus a per-element max-abs-deviation diagnostic.
The Frobenius norm is the standard matrix metric for closeness of
correlation matrices (Higham, 2002, *IMA J. Numer. Anal.* 22:329-343).
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def correlation_residual(
    target: NDArray[np.float64], empirical: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Element-wise residual matrix ``empirical - target``."""
    if target.shape != empirical.shape:
        raise ValueError("target and empirical must share shape")
    return empirical.astype(np.float64) - target.astype(np.float64)


def frobenius_distance(
    target: NDArray[np.float64], empirical: NDArray[np.float64]
) -> dict[str, float]:
    """Frobenius distance and per-element max-abs deviation.

    Parameters
    ----------
    target, empirical : numpy.typing.NDArray[numpy.float64]

    Returns
    -------
    dict[str, float]
        ``frobenius``, ``frobenius_per_dim``, ``max_abs_dev``.
    """
    res = correlation_residual(target, empirical)
    fro = float(np.linalg.norm(res, ord="fro"))
    return {
        "frobenius": round(fro, 6),
        "frobenius_per_dim": round(fro / max(1, target.shape[0]), 6),
        "max_abs_dev": round(float(np.max(np.abs(res))), 6),
    }


def empirical_correlation(df: pd.DataFrame) -> NDArray[np.float64]:
    """Pearson correlation matrix of ``df`` as a ``float64`` array.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    numpy.typing.NDArray[numpy.float64]
    """
    arr = df.select_dtypes(include="number").corr().to_numpy(dtype=np.float64)
    return cast("NDArray[np.float64]", arr)


__all__ = ["correlation_residual", "empirical_correlation", "frobenius_distance"]
