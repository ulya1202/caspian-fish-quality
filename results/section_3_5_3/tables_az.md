# Bölmə 3.5.3 — avtomatik reproduksiya nəticələri

**Seed:** 42 | **n_per_group:** 1000 | **N (ümumi):** 2000

> Mənbə: `python scripts/reproduce_section_3_5_3.py`

## Cədvəl 3.5.6 — Regressiya (5-qat CV, orta R²)

| Hədəf dəyişən | Ridge | Random Forest | LightGBM | XGBoost |
|---|---:|---:|---:|---:|
| Lipid (%) | 0,90 | 0,89 | — | — |
| Body mass (g) | 0,64 | 0,62 | — | — |
| Protein (%) | 0,44 | 0,41 | — | — |

*Qeyd: Hər xana 5-qat kross-validasiyanın orta R²-sidir.*

## Cədvəl 3.5.7 — Zero-shot transfer (Ridge)

| Növ | Hədəf | Proqnoz | Faktiki | Xəta % | Mənbə |
|---|---|---:|---:|---:|---|
| Acipenser stellatus | Lipids (%) | 3,23 | 1,32 | 144,9 | Dorojan et al. (2020) |
| Acipenser stellatus | Protein (%) | 14,06 | 17,86 | -21,3 | Dorojan et al. (2020) |
| Acipenser baerii | Lipids (%) | 2,84 | 5,60 | -49,4 | Lopez et al. (2020) |
| Acipenser baerii | Protein (%) | 15,83 | 17,60 | -10,0 | Lopez et al. (2020) |
| Huso huso | Lipids (%) | 3,19 | 3,92 | -18,6 | Ghomi et al. (2013) |
| Huso huso | Protein (%) | 14,41 | 14,73 | -2,2 | Ghomi et al. (2013) |

**Orta mütləq faiz xətası (Ridge):**
- Protein (%): orta |xəta| = 11.2%; MAPE ≈ 11.2%
- Lipids (%): orta |xəta| = 71.0%; MAPE ≈ 70.9%

## Klassifikasiya (sintetik AG/RG)

- C1_Water→Group: max CV accuracy = 1,0000
- C2_FA→Group: max CV accuracy = 1,0000
- C3_All→Group: max CV accuracy = 1,0000

## Sintetik marginal uyğunluq

- İzlənən parametr sayı: 94
- ≤5% nisbi xəta: 91 / 94
- Maksimum nisbi xəta: 14,21%

## Cədvəl 3.5.8

Bu cədvəl (XGBoost fine-tuning, TVB-N, LOO) notebook-da **yoxdur** — dissertasiya mətnini bu repodakı nəticələrə uyğun yeniləyin və ya gələcək iş kimi qeyd edin.
