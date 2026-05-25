# Limitations

## Synthetic data only

No live fish were sampled. All numbers come from Gaussian-copula (NORTA) sampling of *Silurus glanis* priors extracted from six literature tables (`src/caspian_fish_quality/data/literature/data_1.csv`–`data_6.csv`). Small source studies (often N = 6–50) limit how faithfully marginals and correlations are recovered.

## Cross-species transfer

Models trained on synthetic *S. glanis* are evaluated on published water-quality conditions for *Acipenser stellatus*, *A. baerii*, and *Huso huso* (Dorojan et al., 2020; Lopez et al., 2020; Ghomi et al., 2013). This is **out-of-distribution** transfer across orders (Siluriformes → Acipenseriformes). Lipid predictions are often poor; protein transfer is more stable. Do not treat MAPE as ecological validation.

## Classification on synthetic groups

Aquaculture (AG) vs riverine (RG) labels are separable by construction on synthetic data, so **100% CV accuracy is expected** and is not evidence of real-world discriminability.

## Ethics and CITES

Caspian sturgeons are CITES Appendix II and IUCN threatened. This repository provides a computational substitute, not a protocol for wild sampling.

## Dissertation table 3.5.8

Fine-tuning experiments (TVB-N, storage day, LOO on a small sturgeon set) are **not implemented** in this repository. Regenerate §3.5.6–3.5.7 only via `scripts/reproduce_section_3_5_3.py`.
