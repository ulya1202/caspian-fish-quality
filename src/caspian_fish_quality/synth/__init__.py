"""Synthetic-data generation: moments, truncated normal, Gaussian copula."""

from __future__ import annotations

from caspian_fish_quality.synth.copula import (
    build_corr,
    copula_generate,
    ensure_psd,
    gaussian_copula_sample,
)
from caspian_fish_quality.synth.generators import (
    calc_fa_indices,
    calc_yields,
    generate_all_synthetic,
    get_correlations,
    get_storage_correlations,
    parse_type_a,
    parse_type_b,
    parse_type_c,
    validate,
)
from caspian_fish_quality.synth.moments import sem_to_sd
from caspian_fish_quality.synth.truncated import sample_truncated

__all__ = [
    "build_corr",
    "calc_fa_indices",
    "calc_yields",
    "copula_generate",
    "ensure_psd",
    "gaussian_copula_sample",
    "generate_all_synthetic",
    "get_correlations",
    "get_storage_correlations",
    "parse_type_a",
    "parse_type_b",
    "parse_type_c",
    "sample_truncated",
    "sem_to_sd",
    "validate",
]
