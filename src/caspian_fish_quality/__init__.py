"""caspian_fish_quality: synthetic-data and ML pipeline for Caspian fish quality.

This package supports a PhD dissertation on sturgeon and herring quality in
Azerbaijani Caspian waters. It provides Gaussian-copula NORTA sampling of
literature-derived priors for *Silurus glanis*, leakage-free scikit-learn
pipelines, validation diagnostics, and weak / out-of-distribution transfer
evaluation toward Acipenseriformes.

The public API is intentionally narrow; reach into the subpackages for
finer-grained tooling.
"""

from __future__ import annotations

from caspian_fish_quality.merge.static import merge_static
from caspian_fish_quality.merge.storage import merge_storage
from caspian_fish_quality.ml.models import (
    classification_pipelines,
    regression_pipelines,
)
from caspian_fish_quality.ml.pipeline import build_pipeline
from caspian_fish_quality.synth.copula import (
    build_corr,
    copula_generate,
    ensure_psd,
    gaussian_copula_sample,
)
from caspian_fish_quality.synth.generators import (
    generate_all_synthetic,
    get_correlations,
    get_storage_correlations,
    validate,
)
from caspian_fish_quality.synth.moments import sem_to_sd
from caspian_fish_quality.synth.truncated import sample_truncated
from caspian_fish_quality.transfer.sturgeon_eval import (
    SturgeonReference,
    run_transfer_test,
)

__version__ = "0.1.0"

__all__ = [
    "SturgeonReference",
    "__version__",
    "build_corr",
    "build_pipeline",
    "classification_pipelines",
    "copula_generate",
    "ensure_psd",
    "gaussian_copula_sample",
    "generate_all_synthetic",
    "get_correlations",
    "get_storage_correlations",
    "merge_static",
    "merge_storage",
    "regression_pipelines",
    "run_transfer_test",
    "sample_truncated",
    "sem_to_sd",
    "validate",
]
