from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from caspian_fish_quality.ml.cv import make_cv
from caspian_fish_quality.ml.metrics import (
    classification_metrics,
    regression_metrics,
)
from caspian_fish_quality.ml.models import (
    classification_pipelines,
    regression_pipelines,
)
from caspian_fish_quality.ml.pipeline import build_pipeline


def test_build_pipeline_has_unfitted_scaler_step() -> None:
    pipe = build_pipeline(Ridge())
    scaler = pipe.named_steps["scaler"]
    assert isinstance(scaler, StandardScaler)
    assert not hasattr(scaler, "mean_")


def test_build_pipeline_robust() -> None:
    pipe = build_pipeline(Ridge(), scaler="robust")
    assert pipe.named_steps["scaler"].__class__.__name__ == "RobustScaler"


def test_build_pipeline_none_omits_scaler() -> None:
    pipe = build_pipeline(Ridge(), scaler="none")
    assert "scaler" not in pipe.named_steps


def test_per_fold_scalers_differ() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 3))
    y = rng.standard_normal(50)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    means: list[np.ndarray] = []
    for tr_idx, _ in kf.split(x):
        pipe = build_pipeline(Ridge())
        pipe.fit(x[tr_idx], y[tr_idx])
        means.append(pipe.named_steps["scaler"].mean_.copy())

    for i in range(len(means) - 1):
        assert not np.allclose(means[i], means[i + 1])


def test_make_cv_reg_returns_kfold() -> None:
    cv = make_cv("reg")
    assert cv.__class__.__name__ == "KFold"


def test_make_cv_cls_returns_stratified() -> None:
    cv = make_cv("cls")
    assert cv.__class__.__name__ == "StratifiedKFold"


def test_regression_pipelines_returns_full_catalogue() -> None:
    pipes = regression_pipelines()
    assert "Random Forest" in pipes
    assert "SVR" in pipes
    assert "Linear Regression" in pipes


def test_classification_pipelines_returns_full_catalogue() -> None:
    pipes = classification_pipelines()
    assert "Random Forest" in pipes
    assert "SVM" in pipes


def test_regression_metrics_known_values() -> None:
    out = regression_metrics([1.0, 2.0], [1.0, 2.0])
    assert out["R2"] == 1.0
    assert out["MAE"] == 0.0


def test_classification_metrics_perfect() -> None:
    out = classification_metrics([0, 1, 0], [0, 1, 0])
    assert out["Accuracy"] == 1.0
    assert out["F1"] == 1.0
