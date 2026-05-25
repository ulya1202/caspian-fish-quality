"""Regression and classification scoring summaries."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def regression_metrics(
    y_true: ArrayLike, y_pred: ArrayLike, *, decimals: int = 4
) -> dict[str, float]:
    """Compute R^2, RMSE, MAE.

    Parameters
    ----------
    y_true, y_pred : array-like
    decimals : int, default 4

    Returns
    -------
    dict[str, float]
    """
    return {
        "R2": round(r2_score(y_true, y_pred), decimals),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), decimals),
        "MAE": round(mean_absolute_error(y_true, y_pred), decimals),
    }


def classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    average: str = "weighted",
    decimals: int = 4,
) -> dict[str, float]:
    """Compute accuracy, F1, precision, recall (weighted by default).

    Parameters
    ----------
    y_true, y_pred : array-like
    average : str, default 'weighted'
    decimals : int, default 4

    Returns
    -------
    dict[str, float]
    """
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), decimals),
        "F1": round(f1_score(y_true, y_pred, average=average), decimals),
        "Precision": round(
            precision_score(y_true, y_pred, average=average, zero_division=0),
            decimals,
        ),
        "Recall": round(recall_score(y_true, y_pred, average=average), decimals),
    }


__all__ = ["classification_metrics", "regression_metrics"]
