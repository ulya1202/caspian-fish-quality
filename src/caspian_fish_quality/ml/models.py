"""Standard regression and classification model pipelines.

Hyperparameters mirror the dissertation notebook. Every estimator is wrapped
in a leakage-free :class:`~sklearn.pipeline.Pipeline` via
:func:`caspian_fish_quality.ml.pipeline.build_pipeline`.

XGBoost and LightGBM are optional dependencies; they are imported lazily so
that the package remains importable on platforms where their native runtimes
(e.g. macOS ``libomp``) are not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR

from caspian_fish_quality.ml.pipeline import build_pipeline

if TYPE_CHECKING:  # pragma: no cover
    pass


def _try_import_xgboost() -> Any | None:
    try:
        import xgboost as xgb

        xgb.XGBRegressor(n_estimators=1)
    except Exception:  # pragma: no cover
        return None
    return xgb


def _try_import_lightgbm() -> Any | None:
    try:
        import lightgbm as lgb

        lgb.LGBMRegressor(n_estimators=1)
    except Exception:  # pragma: no cover
        return None
    return lgb


def regression_pipelines(*, random_state: int = 42) -> dict[str, Pipeline]:
    """Build the canonical 9-model regression catalogue.

    XGBoost and LightGBM entries are skipped silently if those backends
    are not importable on the host (e.g. missing ``libomp`` on macOS).

    Parameters
    ----------
    random_state : int, default 42

    Returns
    -------
    dict[str, sklearn.pipeline.Pipeline]
        Display name -> fresh pipeline.
    """
    pipes: dict[str, Pipeline] = {
        "Linear Regression": build_pipeline(LinearRegression()),
        "Ridge": build_pipeline(Ridge(alpha=1.0, random_state=random_state)),
        "Lasso": build_pipeline(Lasso(alpha=0.01, random_state=random_state)),
        "ElasticNet": build_pipeline(
            ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=random_state)
        ),
        "Random Forest": build_pipeline(
            RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=5,
                random_state=random_state,
                n_jobs=-1,
            )
        ),
        "Gradient Boosting": build_pipeline(
            GradientBoostingRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=random_state,
            )
        ),
        "SVR": build_pipeline(SVR(kernel="rbf", C=10.0)),
        "KNN": build_pipeline(KNeighborsRegressor(n_neighbors=7, weights="distance")),
    }
    xgb = _try_import_xgboost()
    if xgb is not None:
        pipes["XGBoost"] = build_pipeline(
            xgb.XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=random_state,
                verbosity=0,
            )
        )
    lgb = _try_import_lightgbm()
    if lgb is not None:
        pipes["LightGBM"] = build_pipeline(
            lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=random_state,
                verbose=-1,
            )
        )
    return pipes


def classification_pipelines(*, random_state: int = 42) -> dict[str, Pipeline]:
    """Build the canonical 7-model classification catalogue.

    XGBoost and LightGBM entries are skipped silently if those backends
    are not importable on the host (e.g. missing ``libomp`` on macOS).

    Parameters
    ----------
    random_state : int, default 42

    Returns
    -------
    dict[str, sklearn.pipeline.Pipeline]
    """
    pipes: dict[str, Pipeline] = {
        "Logistic Regression": build_pipeline(
            LogisticRegression(max_iter=2000, C=1.0, random_state=random_state)
        ),
        "Random Forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1,
            )
        ),
        "Gradient Boosting": build_pipeline(
            GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                random_state=random_state,
            )
        ),
        "SVM": build_pipeline(SVC(kernel="rbf", C=10.0, random_state=random_state)),
        "KNN": build_pipeline(KNeighborsClassifier(n_neighbors=7, weights="distance")),
    }
    xgb = _try_import_xgboost()
    if xgb is not None:
        pipes["XGBoost"] = build_pipeline(
            xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                random_state=random_state,
                verbosity=0,
                eval_metric="logloss",
            )
        )
    lgb = _try_import_lightgbm()
    if lgb is not None:
        pipes["LightGBM"] = build_pipeline(
            lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                random_state=random_state,
                verbose=-1,
            )
        )
    return pipes


__all__ = ["classification_pipelines", "regression_pipelines"]
