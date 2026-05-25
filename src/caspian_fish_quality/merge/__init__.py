"""Mergers that flatten the per-table synthetic dictionary into wide CSVs."""

from __future__ import annotations

from caspian_fish_quality.merge.static import merge_static
from caspian_fish_quality.merge.storage import merge_storage

__all__ = ["merge_static", "merge_storage"]
