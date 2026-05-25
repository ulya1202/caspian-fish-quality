# Demo artifacts (inference-only)

Pre-trained Ridge models and dissertation tables for the Streamlit defense demo.

Regenerate after changing the pipeline, seed, or `n_per_group`:

```bash
python scripts/export_demo_artifacts.py
```

Contents:

- `models/ridge_*.joblib` — fitted pipelines (Protein, Lipids, Body mass)
- `regression_cv_r2.csv` — table 3.5.6
- `transfer_zero_shot.csv`, `transfer_table_3_5_7.csv` — transfer
- `marginal_relative_error.csv` — synthetic vs literature marginals
- `manifest.json` — seed, N, version metadata
