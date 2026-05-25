# Metodologiya

Bu fəsil `caspian_fish_quality` paketində istifadə edilən sintetik
məlumat generasiyasının, maşın öyrənməsi (ML) emal zəncirinin və
validasiya protokolunun mərhələ-mərhələ təsvirini verir.

## 9.1 Sintetik məlumat generasiyası

### 9.1.1 İlk mərhələ: ədəbiyyat priorları

Altı peer-reviewed məqalədən (Bergstrom et al., 2022; Yazici &
Yazicioglu, 2020; Simeanu et al., 2022; Ljubojevic et al., 2013;
Jankowska et al., 2007; Hallier et al., 2007) cədvəl şəklində verilmiş
kütlə-ortası, SEM, min/max və n çıxarılır. SD bərpası Altman & Bland
(2005) düsturu ilə aparılır:

\[
\hat{\sigma} = SEM \cdot \sqrt{n}.
\]

Korrelyasiya priorları cüt-cüt verildiyi üçün sparse mənbə-cüt → r
xəritəsi qurulur və `build_corr` daxilində ən yaxın PSD korrelyasiya
matrisinə (Higham, 2002) yansıdılır.

### 9.1.2 Marginal nümunələmə (kəsilmiş normal)

Hər biokimyəvi/biometrik dəyişən üçün biological cəhətdən etibarlı
[a, b] aralığında kəsilmiş normal paylanma istifadə olunur (Robert, 1995;
Burkardt, 2014). NaN min/max halında defolt olaraq `mean ± 6·sd`
götürülür. SD = 0 halı sabit massiv qaytarır.

### 9.1.3 Birgə nümunələmə (Qauss kopulası, NORTA)

Sklar (1959) teoremi əsasında *k*-ölçülü paylanma **yanlız**
*k* ədəd marginal və *k×k* korrelyasiya matrisi vasitəsilə təmsil
oluna bilər. NORTA addımları (Cario & Nelson, 1997; Ghosh & Henderson,
2003):

1. PSD korrelyasiya matrisi *R*-in Cholesky parçalanması: *R = L Lᵀ*.
2. Müstəqil standart normallar *ε ~ N(0, I)*; *Z = L ε*.
3. Sütun-sütun *U_j = Φ(Z_j)*, sonra *X_j = F_j⁻¹(U_j)* (kəsilmiş normal
   üçün `truncnorm.ppf`).

`np.clip(U, 1e-10, 1 − 1e-10)` ekstremal ucların skipplənməsi üçündür.

## 9.2 Cədvəlləri birləşdirmə

* `merge_static` (Cədvəl 1, 2, 3, 6) — bir balıq bir sətir; sütunlar
  `bio_*`, `idx_*`, `cut_*`, `fa_*` prefiksli.
* `merge_storage` (Cədvəl 4, 5) — `(ölçü, gün)` çoxsətirli pivot →
  `stor_*_dayK` və `chem_*_dayK`.

## 9.3 Maşın öyrənməsi protokolu

### 9.3.1 Sızmasız emal zənciri

`StandardScaler` `sklearn.pipeline.Pipeline` daxilində qoyulur (Pedregosa
et al., 2011); cross-validation hər bölüm üçün scaler-i təzədən fitləşir.
Bu Kapoor & Narayanan (2023) tərəfindən sənədləşdirilmiş ən tez-tez rast
gəlinən leakage formasını dəf edir.

### 9.3.2 Modellər

* **Reqresiya:** Linear, Ridge, Lasso, ElasticNet, Random Forest,
  Gradient Boosting, XGBoost, LightGBM, SVR, KNN.
* **Klassifikasiya:** Logistic Regression, Random Forest, GB, XGBoost,
  LightGBM, SVM, KNN.

Hiperparametrlər notebook-da göstərilənlərlə eynidir; bütün stochastic
addımlar `random_state=42` ilə deterministikdir.

### 9.3.3 Cross-validation

* Reqresiya — `KFold(n_splits=5, shuffle=True, random_state=42)`.
* Klassifikasiya — `StratifiedKFold(...)`. Stone (1974), Kohavi (1995).

### 9.3.4 Metriklər

* Reqresiya — R², RMSE, MAE.
* Klassifikasiya — Accuracy, F1 (weighted), Precision, Recall.

## 9.4 Validasiya

* `ks_per_variable` — Massey (1951), Bonferroni korrelksiyası ilə.
* `mwu_per_variable` — Mann & Whitney (1947).
* `wasserstein_per_variable` — Villani (2009).
* `frobenius_distance` — joint korrelyasiya matrisinin Frobenius
  yaxınlığı (Higham, 2002).
* `tstr_protocol` — Esteban et al. (2017); TRTR-TSTR fərqi.

## 9.5 Cross-species transfer

`run_transfer_test` *S. glanis* üzərində öyrədilmiş modeli üç Caspian
sturgeon növü üçün referans dəyərlərə qarşı sınayır. **Bu metod weak /
out-of-distribution transferdir** (Pan & Yang, 2010); yalnız zəif
uğur kriteriyaları (predicted ↔ observed istiqamət uyğunluğu) tətbiq
olunur.

## 9.6 Domen oxşarlığı

`h_divergence_proxy` source və target rüfləri arasında ikili
klassifikator öyrətməklə Ben-David et al. (2010) çərçivəsindəki
H-divergence-in praktik proxy-ni qaytarır. Balanced accuracy 0.5-ə yaxın
olduqda domenlər oxşardır; 1.0-a yaxın olduqda transfer çətindir.

## 9.7 Reproduktivlik

* Bütün stochastic funksiyalar `numpy.random.Generator` qəbul edir.
* `Settings` (Pydantic v2) master `seed=42` saxlayır.
* `pyproject.toml`-da pinləşmiş asılılıqlar.
* GitHub Actions üzərində 3.10–3.13 matrix.
