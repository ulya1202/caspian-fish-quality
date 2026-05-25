# Dissertation section 3.5.3 — frozen results

These files are produced by:

```bash
python scripts/reproduce_section_3_5_3.py
```

| File | Content |
|------|---------|
| `tables_az.md` | Paste-ready Azerbaijani table snippets (3.5.6, 3.5.7) |
| `regression_cv_r2.csv` | Table 3.5.6 source data (CV R² vs dissertation draft) |
| `transfer_table_3_5_7.csv` | Table 3.5.7 (Ridge zero-shot transfer) |
| `transfer_mape_summary.csv` | Mean absolute % error by target |
| `marginal_relative_error.csv` | Literature mean vs synthetic mean (≤5% claim) |
| `ml_results_all.csv` | Full experiment grid |
| `classification_cv.csv` | Classification experiments |

Default settings: `seed=42`, `n_per_group=1000` (override with `CFQ_SEED`, `CFQ_N_PER_GROUP`).

Table **3.5.8** (fine-tuning) is not generated here — it is not implemented in the canonical notebook.
