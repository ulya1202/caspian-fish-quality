# Məhdudiyyətlər və risklər

Bu fəsil dissertasiyada və `caspian_fish_quality` paketində istifadə
olunan yanaşmaların elmi məhdudiyyətlərini, etika və validasiya ilə bağlı
risklərini siyahıya alır. **Hər hansı elmi nəticəyə əsaslanmadan əvvəl
oxunmalıdır.**

## 5.1 Sintetik məlumat

* **Heç bir canlı orqanizm üzərində birbaşa təcrübə aparılmamışdır.**
  Bütün rəqəmsal nəticələr ədəbiyyatdan çıxarılmış marginallar və
  korrelyasiya priorları üzərində NORTA Qauss kopulası ilə yaradılmış
  sintetik məlumat üzərində alınmışdır.
* Marginal seçimləri (truncated normal) bütün biokimyəvi
  parametrləri yetərincə təsvir etmir (məs. yağ turşuları zərrə-bənzər
  sıfıra yaxın hallara, çəki paylanması isə sağa-əyilməyə meyllidir).
* Korrelyasiya priorları kiçik nümunələrdən (N = 6–50) götürüldüyündən
  Pearson r-ləri özlüyündə geniş güvən intervallarına malikdir;
  Ghosh & Henderson (2003) yüksək ölçülü hallarda NORTA-nın infeasible
  ola biləcəyini göstərir.

## 5.2 Cross-species transfer

* *Silurus glanis* (Siluriformes) → *Acipenser/Huso* (Acipenseriformes)
  keçidi **out-of-distribution / weak transfer** kateqoriyasındadır
  (Pan & Yang, 2010); filogenetik məsafə təxminən 200+ milyon il
  divergent xətti təşkil edir.
* Yalnız direction-of-effect uyğunluğu (zəif uğur kriteriyası) tətbiq
  olunur. MAPE absolyut etibarlılıq üçün istifadə edilməməlidir.
* Florescu Gune et al. (2021) və Dorojan (2020) — bu mənbələr
  Crossref-də 1:1 tapılmadı; alternativ olaraq Bronzi & Rosenthal (2014)
  və Boscari et al. (2017) referans dəyərlər kimi istifadə olunur.
  Modul içindəki dataclass siyahıları yenilənmişdir.

## 5.3 Etika və CITES

* Caspian sturgeon növləri (Huso huso, A. persicus, A. gueldenstaedtii,
  A. nudiventris, A. stellatus) **CITES Appendix II** və IUCN Red List
  CR/EN statusundadır (CITES, 2018; IUCN, 2022).
* Live experimentation institutional review (institutional ethics
  committee) və CITES export permit tələb edir; bu PhD layihəsində belə
  təcrübələr aparılmadığından sintetik yanaşma seçilmişdir.
* Synthetic-data publikasiyası onun real fəaliyyət əvəzinə işləyə
  bilməsi izlənimini yaradacaqsa **vacib disclosure** tələb olunur (bax:
  README, CITATION.cff).

## 5.4 Citasiya etibarlılığı

Mətndə [VERIFY] işarəsi ilə qeyd olunmuş bütün mənbələr Crossref,
PubMed, naşir səhifələri, arXiv və ya institusional URL-lərlə
yoxlanılmışdır. Yoxlanıla bilməyən mənbələr siyahıdan çıxarılaraq
yaxınlıqdakı verifield alternativlərlə əvəz edilmişdir:

| Orijinal mənbə (problem) | Verifield alternativ |
|---|---|
| Florescu Gune et al. (2021) — Crossref-də 1:1 tapılmadı | Bronzi & Rosenthal (2014) — Caspian Sea sturgeons |
| Dorojan (2020) — naşir səhifəsi əlçatmaz | Boscari et al. (2017) — sterlet biokimyəvi xarakteristikası |
| Bergstrom et al. (2022) — orijinal versiya retracted; **erratumlu yenilənmiş** versiya istifadə olunur | post-erratum 2022 versiyası |

## 5.5 Maşın öyrənməsi məhdudiyyətləri

* Yüksək ölçülü kiçik n hallarında ML modellərin variansı yüksəkdir;
  K=5-fold CV bunu tam həll etmir.
* Synthetic-train-and-evaluate-on-synthetic (TSTS) protokolunun nəticələri
  optimist yan tutur; TSTR (Esteban et al., 2017) və domain-check
  (`h_divergence_proxy`, Ben-David et al., 2010) həqiqi qiymətləndirmə
  üçün lazımdır.

## 5.6 Gələcək iş

Bu paketdə **olmayan** və gələcək versiyalarda nəzərdə tutulan istiqamətlər:

* CTGAN / TVAE / vine kopulaları ilə daha güclü generativ alternativlər.
* ANAS (Azərbaycan Milli Elmlər Akademiyası) faktiki məlumat dəstinin
  daxil edilməsi (etika icazələri əldə olunduqdan sonra).
* Real Caspian sturgeon nümunələri ilə ev-əli validasiya.
* Bayesian hierarchical priors (multi-study evidence synthesis).
