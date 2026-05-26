# Correlation priors for Gaussian copula (NORTA)

## Important disclaimer

The bundled literature CSV files (Simeanu et al., 2022) provide **marginal** means, SEM/SD, and min/max per trait. They do **not** publish a full pairwise correlation matrix for all traits. Therefore `get_correlations()` uses **documented structural priors** — plausible Pearson *r* between traits that are biologically coupled — not values transcribed from a single table cell.

For thesis defense, describe this as:

> «Kopula korrelyasiya matrisi Simeanu və b. (2022) cədvəllərindən birbaşa çıxarılmayıb; morfometrik, kəsim yield və yağ turşusu biokimyası üçün ədəbiyyata uyğun struktur priorlar tətbiq olunub (bax: `correlation_priors.yaml`).»

## Method

1. Sparse pairs `(feature_a, feature_b) → r` per source table id (1, 2, 3, 6).
2. `build_corr()` fills missing off-diagonal entries with 0 and diagonal with 1.
3. `ensure_psd()` projects to the nearest PSD matrix (Higham, 2002; see `synth/copula.py`).

Storage time-series (tables 4–5) use `get_storage_correlations()` with priors aligned to Simeanu (2022) Table 4–5 trends (losses ↔ water, proteins ↔ lipids).

## Table 1 — biometrics (data_1)

| Pair (normalized names) | r | Rationale |
|-------------------------|---|-----------|
| bodymass ↔ totallength | 0.95 | Length–weight scaling; Simeanu (2022) discusses allometric growth |
| bodymass ↔ standardlength | 0.94 | Standard length tracks total length (r ≈ 0.98 in paper text for related measures) |
| totallength ↔ standardlength | 0.98 | Near-linear fish body axis |
| Other body dimension pairs | 0.75–0.92 | Positive allometry among head height, circumference, thickness |

**Reference:** Simeanu et al. (2022), Table 1; general morphometric theory (Froese & Pauly, FishBase length–weight concepts).

## Table 2 — condition indices (data_2)

| Pair | r | Rationale |
|------|---|-----------|
| profileindex ↔ fultoncoefficient | −0.60 | Different body shape vs condition metrics |
| fultoncoefficient ↔ qualityindex | 0.70 | Condition-related indices co-vary |
| profileindex ↔ qualityindex | −0.35 | RG vs AG shape differences in Simeanu discussion |

**Reference:** Simeanu et al. (2022), Section 3.2 (Fulton 0.82 RG vs 0.91 AG; profile index 6.00 RG vs 5.44 AG).

## Table 3 — yields (data_3)

| Pair | r | Rationale |
|------|---|-----------|
| livemass ↔ carcassmass | 0.98 | Mass conservation in processing |
| torsomass ↔ filletmass | 0.97 | Cut mass components |
| carcass/torso/fillet yields | 0.50–0.75 | Yield ratios partially coupled |

**Reference:** Simeanu et al. (2022), Table 3; Jankowska et al. (2006) for comparable yield ranges.

## Table 6 — fatty acids (data_6)

| Pair | r | Rationale |
|------|---|-----------|
| c20:5 ↔ c22:6 | 0.70 | LC-PUFA biosynthesis chain |
| c16:0 ↔ c18:2 | −0.35 | SFA vs PUFA trade-off |
| c18:1 ↔ c18:2 | −0.40 | MUFA vs PUFA competition |

**Reference:** Simeanu et al. (2022), Table 6 (AG vs RG MUFA/PUFA balance).

## Storage (tables 4–5)

| Pair | r | Rationale |
|------|---|-----------|
| losses ↔ water | 0.80 | Mass loss tracks moisture change during refrigeration |
| lipids ↔ water | −0.50 | Lipid and water fractions in proximate balance |
| proteins ↔ lipids | −0.35 | Observed diverging trends over storage days in Table 5 |

**Reference:** Simeanu et al. (2022), Tables 4–5 (0–15 days chilled storage).

## Machine-readable source of truth

Edit pairs in:

`src/caspian_fish_quality/data/correlation_priors.yaml`

After edits, run `pytest tests/test_correlations.py` and `python scripts/reproduce_section_3_5_3.py`.
