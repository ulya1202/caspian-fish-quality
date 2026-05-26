# Limitations

## Synthetic data only

No live fish were sampled. Marginal priors come from six CSV tables (`data_1.csv`–`data_6.csv`), all extracted from **Simeanu et al. (2022)** (*Agriculture* 12(12):2144, [DOI 10.3390/agriculture12122144](https://doi.org/10.3390/agriculture12122144)). Copula correlation targets are **structural priors** in `data/correlation_priors.yaml` (see `docs/CORRELATION_PRIORS.md`), not a published full correlation matrix. Small source studies (often N = 6–50 per cell) limit how faithfully marginals and correlations are recovered.

## Cross-species transfer

Models trained on synthetic *S. glanis* are evaluated on published water-quality conditions for *Acipenser stellatus*, *A. baerii*, and *Huso huso* (Dorojan et al., 2020; Lopez et al., 2020; Ghomi et al., 2013). This is **out-of-distribution** transfer across orders (Siluriformes → Acipenseriformes). Lipid predictions are often poor; protein transfer is more stable. Do not treat MAPE as ecological validation.

## Classification on synthetic groups

Aquaculture (AG) vs riverine (RG) labels are separable by construction on synthetic data, so **100% CV accuracy is expected** and is not evidence of real-world discriminability.

## Ethics and CITES

Caspian sturgeons are CITES Appendix II and IUCN threatened. This repository provides a computational substitute, not a protocol for wild sampling.

## Dissertation table 3.5.8

Fine-tuning experiments (TVB-N, storage day, LOO on a small sturgeon set) are **not implemented** in this repository. Regenerate §3.5.6–3.5.7 only via `scripts/reproduce_section_3_5_3.py`.
