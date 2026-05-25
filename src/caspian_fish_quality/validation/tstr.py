"""Train-Synthetic, Test-Real (TSTR) protocol.

Esteban et al. (2017, *RGAN*; arXiv:1706.02633) and Jordon et al. (2019,
*PATE-GAN*) standardised TSTR as the gold-standard utility check for
synthetic data: a model trained only on synthetic inputs must achieve
performance close to one trained on real inputs when both are evaluated
on a held-out real set. We report the gap in the chosen scoring metric.

References
----------
Esteban, C., Hyland, S. L., & Ratsch, G. (2017). Real-valued (medical)
    time series generation with recurrent conditional GANs.
    arXiv:1706.02633.
Jordon, J., Yoon, J., & van der Schaar, M. (2019). PATE-GAN: Generating
    synthetic data with differential privacy guarantees. *ICLR*.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from caspian_fish_quality.ml.metrics import (
    classification_metrics,
    regression_metrics,
)
from caspian_fish_quality.ml.models import (
    classification_pipelines,
    regression_pipelines,
)

Task = Literal["reg", "cls"]


def tstr_protocol(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    target: str,
    features: Iterable[str],
    *,
    task: Task = "reg",
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare TRTR vs TSTR for every model in the catalogue.

    Parameters
    ----------
    real, synthetic : pandas.DataFrame
        Must share at least the ``features`` columns and ``target``.
    target : str
        Name of the target column.
    features : iterable of str
    task : {'reg', 'cls'}, default 'reg'
    test_size : float, default 0.2
    random_state : int, default 42

    Returns
    -------
    pandas.DataFrame
        One row per model with TRTR / TSTR scores plus the gap.
    """
    feats = [f for f in features if f in real.columns and f in synthetic.columns]
    if not feats:
        raise ValueError("No common features between real and synthetic")

    x_real = real[feats].to_numpy()
    y_real = real[target].to_numpy()
    x_syn = synthetic[feats].to_numpy()
    y_syn = synthetic[target].to_numpy()

    x_tr, x_te, y_tr, y_te = train_test_split(
        x_real, y_real, test_size=test_size, random_state=random_state
    )

    pipes: dict[str, Pipeline] = (
        regression_pipelines(random_state=random_state)
        if task == "reg"
        else classification_pipelines(random_state=random_state)
    )
    metric_key = "R2" if task == "reg" else "Accuracy"
    rows: list[dict[str, float | str]] = []

    for name, pipe in pipes.items():
        pipe_real: Pipeline = pipe
        pipe_real.fit(x_tr, y_tr)
        y_real_pred = pipe_real.predict(x_te)
        if task == "reg":
            real_score = regression_metrics(y_te, y_real_pred)[metric_key]
        else:
            real_score = classification_metrics(y_te, y_real_pred)[metric_key]

        pipe_syn = (
            regression_pipelines(random_state=random_state)
            if task == "reg"
            else classification_pipelines(random_state=random_state)
        )[name]
        pipe_syn.fit(x_syn, y_syn)
        y_syn_pred = pipe_syn.predict(x_te)
        if task == "reg":
            syn_score = regression_metrics(y_te, y_syn_pred)[metric_key]
        else:
            syn_score = classification_metrics(y_te, y_syn_pred)[metric_key]

        rows.append(
            {
                "model": name,
                "TRTR": float(real_score),
                "TSTR": float(syn_score),
                "gap": round(float(real_score) - float(syn_score), 4),
            }
        )

    return pd.DataFrame(rows)


__all__ = ["Task", "tstr_protocol"]
