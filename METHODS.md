# Methods (English summary)

This document summarises the methods implemented in `caspian_fish_quality`.
For the full Azerbaijani methods chapter, see
[`docs/az/methods.md`](docs/az/methods.md).

## 1. Synthetic data generation

* **Marginal recovery.** SD is reconstructed from SEM via
  `sem_to_sd(sem, n) = sem * sqrt(n)` (Altman & Bland, 2005).
* **Truncated-normal sampling.** Each biologically-bounded variable is
  drawn from a truncated normal distribution using the optimal
  exponential-proposal accept-reject sampler of Robert (1995); see
  Burkardt (2014) for practical formulae and Johnson, Kotz & Balakrishnan
  (1994) for moments.
* **Joint sampling (NORTA).** The NORmal-To-Anything construction
  (Cario & Nelson, 1997) draws correlated multivariate normals from the
  Cholesky factor of a target Pearson correlation matrix, applies the
  standard normal CDF to obtain uniforms in (0, 1), then maps each
  column through the inverse marginal CDF. Theoretical foundation:
  Sklar's theorem (Sklar, 1959; Nelsen, 2006).
* **Correlation projection.** Target correlation matrices are projected
  onto the nearest positive-semidefinite correlation matrix via
  ``statsmodels.stats.correlation_tools.corr_nearest`` (Higham, 2002),
  with eigenvalue clipping fallback.

## 2. Data engineering

* `eda.parsing`: `divide_by_symbol`, `clean_scientific_notation`,
  `auto_clean_values`, `clean_data`, `coerce_numeric` (replaces deprecated
  `pd.to_numeric(errors='ignore')`).
* `eda.regex_extract`: per-column ``±`` splitter; ``n=<int>``
  detection.
* `synth.generators`: parsers for three table layouts (A: per-trait
  Mean/SEM with optional Min/Max; B: time-series storage; C: fatty-acid
  composition).

## 3. Machine-learning protocol

* All estimators wrapped in `sklearn.pipeline.Pipeline` with
  `StandardScaler` -> model -> protect against preprocessing leakage
  (Kapoor & Narayanan, 2023; Kuhn & Johnson, 2013).
* 5-fold KFold (regression) or StratifiedKFold (classification);
  Stone (1974), Kohavi (1995).
* Catalogue: Linear, Ridge, Lasso, ElasticNet, Random Forest,
  Gradient Boosting, XGBoost, LightGBM, SVR/SVC, KNN.
* Metrics: R^2, RMSE, MAE for regression; accuracy, F1 (weighted),
  precision, recall for classification.

## 4. Validation

* `validation.marginal`: Kolmogorov-Smirnov (Massey, 1951) and
  Mann-Whitney U (Mann & Whitney, 1947) per variable, with optional
  Bonferroni correction (Bonferroni, 1936).
* `validation.joint`: Frobenius distance between target and empirical
  Pearson correlation matrices, plus per-element max-abs deviation.
* `validation.wasserstein`: 1-Wasserstein distance per variable
  (Villani, 2009).
* `validation.tstr`: Train-Synthetic, Test-Real protocol (Esteban et
  al., 2017; Jordon et al., 2019); reports the per-model TRTR-TSTR gap.

## 5. Cross-species transfer

* `transfer.sturgeon_eval`: train donor models on synthetic *Silurus
  glanis* data, zero-shot predict for *A. stellatus*, *A. baerii*, and
  *H. huso* using published water-quality conditions (Dorojan et al.,
  2020; Lopez et al., 2020; Ghomi et al., 2013). Reports MAPE and
  percent error vs literature proximate composition.
* `transfer.domain_check`: H-divergence proxy via balanced 5-fold CV
  on a logistic-regression domain classifier (Ben-David et al., 2010).

## 6. Reproducibility

* All stochastic functions take `numpy.random.Generator`.
* `Settings` (Pydantic v2) carries the master seed.
* Versions pinned in `pyproject.toml`.
* GitHub Actions matrix: Python 3.10, 3.11, 3.12, 3.13.

For full BibTeX of every reference, see `REFERENCES.bib`.
