# Ədəbiyyat icmalı

Bu fəsil Xəzər dənizində nərəkimilər (Acipenseriformes) və kilka
(*Clupeonella* spp.) keyfiyyətinə dair çoxsahəli ədəbiyyatı qısa şəkildə
cəmləyir. Sintetik məlumat hissəsində istifadə edilən bütün rəqəmsal
priorlar (mean, SD, korrelyasiya) peer-reviewed mənbələrdən çıxarılıb;
tam BibTeX siyahısı `REFERENCES.bib` faylındadır.

## 2.1 Xəzər ekosistemi və balıq sərvəti

Xəzər dünyanın ən böyük qapalı su hövzəsidir; nərə populyasiyaları üçün
qlobal əhəmiyyətə malik unikal məkandır (Lagutov, 2008; Pourkazemi, 2006).
Sovet sonrası dövrdə qaçaqmal ovu, hidrotexniki tikintilər (Volqada
suboruları), və neft-qaz çirklənməsi populyasiyaları kritik səviyyəyə
endirib (Friedrich, 2018; Pikitch et al., 2005). Mamedov (2006, *ICES
Journal of Marine Science*) Xəzərin Azərbaycan zonasında nərə miqrasiya
modellərini sənədləşdirir; bu işin nəticələri canlı təcrübə əvəzinə
ədəbiyyatlı sintez yanaşmasının zəruriliyini göstərir.

## 2.2 *Huso huso* (bölgə / ağ balıq) biologiyası

*Huso huso* dünyanın ən böyük şirin-su balığıdır; CITES Appendix II və
IUCN Red List CR statusundadır (CITES, 2018; IUCN, 2022). Yetişkin
fərdlərin orta çəkisi 80-300 kq, uzunluğu 2-4 m olur (Bronzi & Rosenthal,
2014; Boscari et al., 2017). Azərbaycan mətbəxində **bölgə** (regional)
adlandırılır; bəzi mətbəx mənbələrində **ağ balıq** ilə qeyd olunur,
lakin elmi ədəbiyyatda **uzunburun** termini *A. nudiventris* üçün
saxlanılır (Kottelat & Freyhof, 2007). **Bu dissertasiyada terminoloji
qarışıqlıq olmaması üçün *Huso huso* yalnız "bölgə" yazılır.**

## 2.3 *Silurus glanis* (naqqa balığı) — donor növ

*Silurus glanis* sintetik məlumat priorlarımızın əsas mənbəyidir.
Bergstrom et al. (2022, *Aquaculture Reports*) və Yazici & Yazicioglu
(2020) Avropa naqqa balığının biometrik və biokimyəvi göstəricilərini
müqayisəli ədəbiyyatda təqdim edir. Simeanu et al. (2022) yetişdirmə
şəraitində protein və lipid faizini, Ljubojevic et al. (2013) yağ
turşuları profili (n-3, n-6, AI/TI indeksləri) verir. Jankowska et al.
(2007) qida növünün lipid kompozisiyasına təsirini, Hallier et al.
(2007) saxlama stabilliyini bildirir.

## 2.4 Yağ turşuları və keyfiyyət indeksləri

PUFA/SFA, n-3/n-6, atherogenic (AI) və thrombogenic (TI) indeksləri
Ulbricht & Southgate (1991, *The Lancet*) tərəfindən təklif olunmuşdur
və balıq məhsulu keyfiyyətini qiymətləndirməkdə standart hala gəlmişdir.
Wood et al. (2008) bu indekslərin müxtəlif balıq növlərində tətbiqini
müqayisə edir.

## 2.5 Kilka *Clupeonella* spp.

*Clupeonella cultriventris caspia*, *C. engrauliformis* və *C. grimmi*
Xəzərin endemik kiçik pelagik balıqlarıdır; ictimai balıqçılığın əsas
hədəfidir (Sedov et al., 2003; Karpyuk et al., 2007). Son onilliklərdə
*Mnemiopsis leidyi* (gel medusa) işğalı populyasiyaları kəskin azaldıb
(Kideys, 2002).

## 2.6 Sintetik məlumat və kopula metodları

Sklar (1959) hər çoxölçülü paylanmanın marginalları və kopulası
vasitəsilə təmsil oluna biləcəyini sübut etdi. Cario & Nelson (1997)
NORTA (NORmal-To-Anything) konstruksiyasını təqdim etdi; Ghosh &
Henderson (2003) yüksək ölçülərdə bu metodun infeasibility hallarını
göstərdi. Higham (2002) PSD-yaxın korrelyasiya matrisinin Frobenius
mənasında ən yaxın PSD üzərinə proyeksiyasını verdi. Robert (1995)
kəsilmiş normal nümunələmənin optimal eksponensial proposal alqoritmini
təqdim etdi. Burkardt (2014) bunun əməli formullarını verdi.

## 2.7 Maşın öyrənməsi və sızma riski

Kapoor & Narayanan (2023, *Patterns*) elm üzrə 294 ML məqalənin
audit-ində preprocessing leakage-in geniş yayıldığını göstərdi.
Kuhn & Johnson (2013) və Pedregosa et al. (2011) `Pipeline` daxilində
scaler-i qoruyaraq qarşı tədbirin standartını verdi. Stone (1974) və
Kohavi (1995) çapraz-validasiyanın əsasını qoydu.

## 2.8 Cross-species transfer

Pan & Yang (2010) transfer öyrənmənin canonical taksonomiyasını verdi.
Ben-David et al. (2010) H-divergence ilə domen oxşarlığının
qiymətləndirilməsi üçün PAC çərçivəsini qurdu. Bizim halda *Silurus
glanis* (Siluriformes) → *Acipenser/Huso* (Acipenseriformes) keçidi
**out-of-distribution / weak transfer** kateqoriyasına aiddir; nəticələr
istiqamət-uyğunluğu (direction match) baxımından şərh olunur.

## 2.9 Validasiya statistikası

Massey (1951) iki-nümunəli Kolmogorov-Smirnov testini, Mann & Whitney
(1947) U statistikasını təqdim etdi. Bonferroni (1936) çoxlu müqayisə
korreksiyasını göstərdi. Esteban et al. (2017) və Jordon et al. (2019)
sintetik məlumat üçün Train-Synthetic, Test-Real (TSTR) protokolunu
standartlaşdırdı. Villani (2009) optimal nəqliyyat və 1-Wasserstein
məsafəsinin nəzəri əsaslarını verdi.

---

Tam BibTeX siyahısı (80+ giriş): `REFERENCES.bib`.
