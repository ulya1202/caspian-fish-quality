# Quickstart: from priors to transfer results

This 80-line walkthrough takes you from the bundled six literature CSVs
to a transfer evaluation table written to ``transfer_results.csv``.

## 1. Load source tables

```python
from caspian_fish_quality.synth.generators import (
    generate_all_synthetic,
    load_default_df_dict,
)

df_dict = load_default_df_dict()
print({k: v.shape for k, v in df_dict.items()})
```

## 2. Generate synthetic *Silurus glanis* data

```python
synthetic = generate_all_synthetic(df_dict, n_per_group=1000, seed=42, verbose=True)
```

`synthetic` is `dict[int, pandas.DataFrame]`; one entry per source table.

## 3. Wide-format merging

```python
from caspian_fish_quality.merge import merge_static, merge_storage

static_df = merge_static(synthetic)
storage_df = merge_storage(synthetic)
```

## 4. Build the PHD water-quality dataset and full feature matrix

```python
from caspian_fish_quality.ml.datasets import (
    generate_phd_dataset,
    prepare_full_dataset,
    run_all,
)

phd_df = generate_phd_dataset(n_per_group=1000, seed=42)
full_df = prepare_full_dataset(phd_df, static_df, storage_df)
results, importance = run_all(full_df, verbose=True, random_state=42)
results.to_csv("ml_results_v3.csv", index=False)
importance.to_csv("feature_importance_v3.csv", index=False)
```

## 5. Validate marginals and joint structure

```python
import pandas as pd
from caspian_fish_quality.validation import (
    ks_per_variable,
    wasserstein_per_variable,
    frobenius_distance,
)
from caspian_fish_quality.validation.joint import empirical_correlation

real = pd.read_csv("silurus_glanis_phd_dataset_v2 (1).csv")
ks = ks_per_variable(real, phd_df, adjust="bonferroni")
w1 = wasserstein_per_variable(real, phd_df)

target_R = empirical_correlation(real.select_dtypes(include="number"))
emp_R = empirical_correlation(phd_df.select_dtypes(include="number"))
fro = frobenius_distance(target_R, emp_R)
```

## 6. Run cross-species transfer to Caspian sturgeons

```python
from caspian_fish_quality.transfer import run_transfer_test

transfer = run_transfer_test(phd_df, static_df, storage_df)
transfer.to_csv("transfer_results.csv", index=False)
```

The output has 27 rows (3 sturgeon species x 3 targets x 3 models). See
{doc}`../az/limitations` for caveats on out-of-distribution transfer.
