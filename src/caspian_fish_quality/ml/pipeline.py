"""Leakage-free preprocessing + estimator pipelines.

`StandardScaler` is wrapped inside `sklearn.pipeline.Pipeline` so that
fitting occurs *only* on each cross-validation training fold, never on
the full dataset. This eliminates the most common form of preprocessing
leakage catalogued by Kapoor & Narayanan (2023), who reviewed 294 ML-in-
science papers and found leakage in a substantial fraction. Kuhn &
Johnson (2013, Ch. 4) provide the canonical applied-statistics treatment.

References
----------
Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility
    crisis in machine-learning-based science. *Patterns*, 4(9), 100804.
    https://doi.org/10.1016/j.patter.2023.100804
Kuhn, M., & Johnson, K. (2013). *Applied Predictive Modeling*. Springer.
    https://doi.org/10.1007/978-1-4614-6849-3
Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python.
    *JMLR*, 12, 2825-2830.
"""

from __future__ import annotations

from typing import Literal

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

ScalerName = Literal["standard", "robust", "none"]


def build_pipeline(estimator: BaseEstimator, scaler: ScalerName = "standard") -> Pipeline:
    """Wrap ``estimator`` in a leakage-free :class:`~sklearn.pipeline.Pipeline`.

    The scaler is constructed *fresh* every call. When the returned pipeline
    is passed to ``cross_val_score`` (or any other CV utility), scikit-learn
    fits it on each training fold only, preventing test-fold information
    from leaking into preprocessing statistics.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        Any fit/predict-compatible scikit-learn estimator.
    scaler : {'standard', 'robust', 'none'}, default 'standard'
        Pre-estimator scaling step. ``'none'`` returns a one-step pipeline.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A fresh pipeline whose preprocessing steps are unfitted.
    """
    if scaler == "standard":
        return Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    if scaler == "robust":
        return Pipeline([("scaler", RobustScaler()), ("model", estimator)])
    if scaler == "none":
        return Pipeline([("model", estimator)])
    raise ValueError(f"Unknown scaler: {scaler!r}")


__all__ = ["ScalerName", "build_pipeline"]
