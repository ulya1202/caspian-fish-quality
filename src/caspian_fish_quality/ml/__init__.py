"""Leakage-free ML pipelines, models, cross-validation, and metrics."""

from __future__ import annotations

from caspian_fish_quality.ml.cv import make_cv
from caspian_fish_quality.ml.datasets import (
    generate_phd_dataset,
    prepare_full_dataset,
    run_all,
    run_experiment,
)
from caspian_fish_quality.ml.metrics import (
    classification_metrics,
    regression_metrics,
)
from caspian_fish_quality.ml.models import (
    classification_pipelines,
    regression_pipelines,
)
from caspian_fish_quality.ml.pipeline import build_pipeline

__all__ = [
    "build_pipeline",
    "classification_metrics",
    "classification_pipelines",
    "generate_phd_dataset",
    "make_cv",
    "prepare_full_dataset",
    "regression_metrics",
    "regression_pipelines",
    "run_all",
    "run_experiment",
]
