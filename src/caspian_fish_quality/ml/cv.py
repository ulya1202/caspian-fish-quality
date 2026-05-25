"""Cross-validation factories.

Five-fold CV follows Stone (1974) and Kohavi (1995). Stratified folds are
used for classification to preserve class balance per fold.

References
----------
Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy
    estimation and model selection. *IJCAI*, 1137-1143.
Stone, M. (1974). Cross-validatory choice and assessment of statistical
    predictions. *J. R. Stat. Soc. B*, 36(2), 111-147.
    https://doi.org/10.1111/j.2517-6161.1974.tb00994.x
"""

from __future__ import annotations

from typing import Literal

from sklearn.model_selection import KFold, StratifiedKFold

Task = Literal["reg", "cls"]


def make_cv(task: Task = "reg", n_splits: int = 5, seed: int = 42) -> KFold | StratifiedKFold:
    """Return the canonical 5-fold ``KFold`` / ``StratifiedKFold``.

    Parameters
    ----------
    task : {'reg', 'cls'}, default 'reg'
        ``'reg'`` returns ``KFold``; ``'cls'`` returns ``StratifiedKFold``.
    n_splits : int, default 5
    seed : int, default 42

    Returns
    -------
    sklearn.model_selection.KFold or sklearn.model_selection.StratifiedKFold
    """
    if task == "reg":
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if task == "cls":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    raise ValueError(f"Unknown task: {task!r}")


__all__ = ["Task", "make_cv"]
