# Literature sources and data provenance

This document maps every bundled input in `src/caspian_fish_quality/data/literature/` to peer-reviewed sources with DOIs. It is the audit trail required for dissertation §3.5.3 and thesis defense.

## Primary source for `data_1.csv` … `data_6.csv`

All six CSV files are **extracted summary tables** from one paper (not six independent studies):

| File | Simeanu et al. (2022) table | Content |
|------|-----------------------------|---------|
| `data_1.csv` | Table 1 | Main biometric traits (AG/RG, n = 50) |
| `data_2.csv` | Calculated indices | Profile, Fulton, quality, thickness, fleshy indices |
| `data_3.csv` | Slaughter / cut yields | Live mass, carcass, torso, fillet mass and yields |
| `data_4.csv` | Table 4 | Refrigerated storage: losses (%) and water (%) |
| `data_5.csv` | Table 5 | Refrigerated storage: ash, proteins, lipids (%) |
| `data_6.csv` | Table 6 | Fatty acid profile (g/100 g FAME) and sanogenic indices |

**Full citation**

> Simeanu, C., Măgdici, E., Păsărin, B., Avarvarei, B.-V., & Simeanu, D. (2022). Quantitative and qualitative assessment of European catfish (*Silurus glanis*) flesh. *Agriculture*, *12*(12), 2144. https://doi.org/10.3390/agriculture12122144

**Open access:** https://www.mdpi.com/2077-0472/12/12/2144

**Species:** *Silurus glanis* (European wels catfish). Groups: **AG** = aquaculture (farm ponds), **RG** = Prut River (natural).

**Extraction note:** Values in CSV match the published tables (e.g. body mass RG 1784.91 ± 37.43 g; AG 1840.71 ± 30.25 g). SEM columns are converted to SD via `sd = sem × sqrt(n)` in `synth/moments.py` (Altman & Bland, 2005).

---

## Secondary references (mechanistic / comparison, not CSV extraction)

These papers support **interpretation**, **PHD water→flesh coefficients**, or **sturgeon transfer benchmarks** — they are **not** the source of the six copula margin tables unless stated otherwise.

### Hallier et al. (2007) — farming temperature and fillet quality

> Hallier, A., Serot, T., & Prost, C. (2007). Influence of farming conditions on colour and texture of European catfish (*Silurus glanis*) flesh. *Journal of the Science of Food and Agriculture*, *87*(5), 739–745. https://doi.org/10.1002/jsfa.2779

Also: Hallier, A., Serot, T., & Prost, C. (2007). Influence of rearing conditions and feed on the biochemical composition of fillets of the European catfish (*Silurus glanis*). *Food Chemistry*, *103*(3), 808–815. https://doi.org/10.1016/j.foodchem.2006.09.021

**Use in repo:** `generate_phd_dataset()` — higher water temperature associated with higher lipid in the synthetic water-quality layer (qualitative direction per Hallier, 2007).

### Jankowska et al. (2006) — slaughter yield comparison

> Jankowska, B., Zakes, Z., Zmijewski, T., Ulikowski, D., & Kowalska, A. (2006). Slaughter value and flesh characteristics of European catfish (*Silurus glanis*) fed natural and formulated feed under different rearing conditions. *European Food Research and Technology*, *224*, 453–459. https://doi.org/10.1007/s00217-006-0348-9

**Use in repo:** Cited in Simeanu (2022) for yield intervals; supports `Feed_Type` direction in PHD generator (farm vs natural feeding).

### Huss (1995) — fish quality assessment

> Huss, H. H. (1995). *Quality and quality changes in fresh fish* (FAO Fisheries Technical Paper 348). Food and Agriculture Organization of the United Nations. https://www.fao.org/4/v7180e/v7180e00.htm

**Use in repo:** Dissolved oxygen → protein direction in PHD generator (general aquaculture water-quality guidance).

### Sturgeon transfer (zero-shot evaluation)

| Species | Water quality | Flesh composition |
|---------|---------------|-------------------|
| *Acipenser stellatus* | Florescu (Gune) et al. (2021) — RAS context, Galați | Dorojan (2016) thesis / Dorojan et al. — growth performance |
| *Acipenser baerii* | Facility estimates (Lopez et al., 2020 context) | Lopez, J., et al. (2020) — see `transfer/sturgeon_eval.py` |
| *Huso huso* | RAS standards (Ghomi et al., 2013 context) | Ghomi, S. M., et al. (2013) |

**Florescu (Gune) et al. (2021)**

> Florescu (Gune), I. E., Georgescu, S. E., Dudu, A., Balas, M., Voicu, S., Grecu, I., Dediu, L., Dinischiotu, A., & Costache, M. (2021). Oxidative stress and antioxidant defense mechanisms in response to starvation and refeeding in the intestine of stellate sturgeon (*Acipenser stellatus*) juveniles from aquaculture. *Animals*, *11*(1), 76. https://doi.org/10.3390/ani11010076

**Ghomi et al. (2013)** — beluga proximate (used for *H. huso* flesh benchmark)

> Ghomi, S. M., Nazari, R. M., & Safari, R. (2013). Effect of dietary supplementation of vitamin E, soybean lecithin and L-carnitine on growth and chemical composition of beluga (*Huso huso*) juveniles. *Journal of Applied Ichthyology*, *29*(3), 596–601. https://doi.org/10.1111/jai.12057

**Lopez et al. (2020)** — Siberian sturgeon composition (verify exact author list in your bibliography)

> Search: *Acipenser baerii* body composition aquaculture 2020 — align with `Lopez et al. (2020)` values in `sturgeon_eval.py` (lipids 5.6%, protein 17.6%).

**Dorojan (2016)** — stellate sturgeon RAS thesis (Galați); repo label “Dorojan et al. (2020)” should be harmonized to the publication you cite in the dissertation.

> Dorojan, O.-G. (2016). *Optimizarea performanței creșterii puietului unor specii de sturioni în condițiile unui ecosistem recirculant de acvacultură* [Doctoral dissertation]. “Dunărea de Jos” University of Galați.

---

## Correlation priors (Gaussian copula)

Pearson correlation targets for NORTA sampling are **not** copied cell-by-cell from Simeanu (2022) (the paper reports few pairwise *r* values). They are **structural priors** documented in:

- `src/caspian_fish_quality/data/correlation_priors.yaml`
- `docs/CORRELATION_PRIORS.md`

Rationale: morphometric allometry (length ↔ mass), yield mass consistency, and fatty-acid biochemistry (SFA/MUFA/PUFA trade-offs) following patterns described in Simeanu Table 6 and standard fish lipid literature.

---

## What this repo does **not** contain

- No field samples from Caspian beluga or kilka (dissertation lab work is separate).
- No TVB-N transfer learning (dissertation table 3.5.8) — see `LIMITATIONS.md`.
- **Kilka** (*Clupeonella*) is not modelled in code; only *S. glanis* synthetic + sturgeon transfer.

---

## How to cite this software

See `CITATION.cff` and:

> Aliyeva, U. (2026). *caspian_fish_quality* (Version 0.1.3) [Computer software]. https://github.com/ulya1202/caspian-fish-quality
