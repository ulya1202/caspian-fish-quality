"""H-divergence proxy for source/target distribution similarity.

Implements the proxy of Ben-David et al. (2010, *Mach. Learn.* 79:151-175):
train a binary classifier to discriminate source from target rows; the
balanced classification error then proxies the H-divergence. A perfect
discriminator (error = 0) implies the domains are completely separable
(transfer is unlikely to help); error close to 0.5 implies the domains
look nearly identical.

References
----------
Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F., &
    Vaughan, J. W. (2010). A theory of learning from different domains.
    *Machine Learning*, 79(1-2), 151-175.
    https://doi.org/10.1007/s10994-009-5152-4
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def h_divergence_proxy(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    """Compute a balanced H-divergence proxy via CV on a domain classifier.

    Parameters
    ----------
    source, target : pandas.DataFrame
        Frames sharing a common set of numeric columns; intersection is
        used.
    n_splits : int, default 5
    random_state : int, default 42

    Returns
    -------
    dict[str, float]
        Keys: ``balanced_accuracy_mean``, ``balanced_accuracy_std``,
        ``h_divergence_proxy``. The H-divergence proxy is defined as
        ``2 * (BA - 0.5)``; values close to 0 indicate similar domains.
    """
    common = sorted(set(source.columns) & set(target.columns))
    common = [
        c
        for c in common
        if pd.api.types.is_numeric_dtype(source[c]) and pd.api.types.is_numeric_dtype(target[c])
    ]
    if not common:
        raise ValueError("No common numeric columns between source and target")

    x_src = source[common].to_numpy(dtype=np.float64)
    x_tgt = target[common].to_numpy(dtype=np.float64)
    x = np.vstack([x_src, x_tgt])
    y = np.concatenate([np.zeros(len(x_src), dtype=np.int64), np.ones(len(x_tgt), dtype=np.int64)])

    pipe: Pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=random_state)),
        ]
    )
    scores = cross_val_score(pipe, x, y, cv=n_splits, scoring="balanced_accuracy")
    ba_mean = float(scores.mean())
    return {
        "balanced_accuracy_mean": round(ba_mean, 4),
        "balanced_accuracy_std": round(float(scores.std()), 4),
        "h_divergence_proxy": round(2.0 * (ba_mean - 0.5), 4),
    }


__all__ = ["h_divergence_proxy"]
