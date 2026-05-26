"""Load documented Pearson correlation priors from ``correlation_priors.yaml``.

See ``docs/CORRELATION_PRIORS.md`` and ``docs/LITERATURE_SOURCES.md``.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


def _parse_pairs(raw: list[list[Any]]) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for entry in raw:
        if len(entry) != 3:
            raise ValueError(f"Expected [feature_a, feature_b, r], got {entry!r}")
        a, b, r = str(entry[0]), str(entry[1]), float(entry[2])
        out[(a, b)] = r
    return out


@lru_cache(maxsize=1)
def load_correlation_priors() -> dict[str, Any]:
    """Return parsed YAML (cached)."""
    root = resources.files("caspian_fish_quality").joinpath("data")
    with resources.as_file(root.joinpath("correlation_priors.yaml")) as path:
        text = Path(path).read_text(encoding="utf-8")
    return yaml.safe_load(text)


def get_correlations() -> dict[int, dict[tuple[str, str], float]]:
    """Pearson correlation priors per literature table id (1, 2, 3, 6)."""
    data = load_correlation_priors()
    tables = data.get("tables", {})
    out: dict[int, dict[tuple[str, str], float]] = {}
    for key, block in tables.items():
        pairs = _parse_pairs(block.get("pairs", []))
        out[int(key)] = pairs
    return out


def get_storage_correlations() -> dict[tuple[str, str], float]:
    """Correlation priors for storage stability columns (tables 4–5)."""
    data = load_correlation_priors()
    block = data.get("storage", {})
    return _parse_pairs(block.get("pairs", []))


__all__ = ["get_correlations", "get_storage_correlations", "load_correlation_priors"]
