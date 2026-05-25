"""Tests for committed demo artifact bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from caspian_fish_quality.demo import load_demo_artifacts


def test_demo_artifacts_load_and_predict() -> None:
    root = Path(__file__).resolve().parents[1]
    art = load_demo_artifacts(root / "demo_artifacts")
    assert len(art.models) == 3
    assert len(art.feat_cols) >= 8
    x = np.array([[0.0] * len(art.feat_cols)])
    for pipe in art.models.values():
        pred = pipe.predict(x)
        assert pred.shape == (1,)
