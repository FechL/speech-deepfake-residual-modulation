# Tabel Bab 4 — residual-modulasi v9

Sumber: run utama `1d5ba7b2b9562c28`, stress `1d5ba7b2b9562c28` (subset bba88e5ae911…), ablation `52760a671e6853be`. Checksum `complete.json` terverifikasi: {'1d5ba7b2b9562c28': {'files': 64, 'known_mismatch': ['bootstrap_intervals.csv', 'error_analysis.csv', 'evidence_consistency.csv', 'metrics.csv', 'seed_summary.csv', 'utterance_scores.csv']}, '52760a671e6853be': {'files': 64, 'known_mismatch': []}}. Desimal memakai koma. Angka dibangkitkan oleh `scripts/build_bab4_assets.py`; jangan diedit manual.

## Tabel 4.1 Komposisi data, cakupan pengukuran, dan durasi

| Dataset | Split | Kelas | n | Residual terukur (%) | Modulasi terukur (%) | Median durasi dianalisis (s) | Median hening terpangkas (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| asvspoof2019_la | dev | REAL | 500 | 100,0 | 100,0 | 1,82 | 1,42 |
| asvspoof2019_la | dev | FAKE | 1.500 | 100,0 | 100,0 | 2,16 | 0,68 |
| asvspoof2019_la | eval | REAL | 1.675 | 100,0 | 100,0 | 1,75 | 1,35 |
| asvspoof2019_la | eval | FAKE | 15.600 | 100,0 | 100,0 | 2,16 | 0,34 |
| asvspoof2019_la | train | REAL | 2.580 | 100,0 | 100,0 | 1,93 | 1,25 |
| asvspoof2019_la | train | FAKE | 22.800 | 100,0 | 100,0 | 2,12 | 0,69 |
| asvspoof2021_df | external | REAL | 187 | 100,0 | 100,0 | 2,37 | 0,74 |
| asvspoof2021_df | external | FAKE | 625 | 100,0 | 100,0 | 2,23 | 0,38 |
| mlaad_tiny | external | REAL | 953 | 100,0 | 100,0 | 6,44 | 0,98 |
| mlaad_tiny | external | FAKE | 953 | 100,0 | 100,0 | 6,34 | 0,30 |

Hening terpangkas = durasi asli − durasi setelah trim 35 dB. Semua 47.373 utterance terekstraksi (`analysis_status = ok`).

## Tabel 4.2 Perbedaan fitur residual dan modulasi antara REAL dan FAKE (LA train)

| Fitur | Median REAL | Median FAKE | Rank-biserial (FAKE − REAL) | q (BH-FDR) |
| --- | --- | --- | --- | --- |
| res_energy_mean | 0,003382 | 0,00217 | −0,349 | < 0,001 |
| res_entropy_mean | 4,416 | 4,329 | −0,301 | < 0,001 |
| res_prediction_error_mean | 0,0777 | 0,06688 | −0,210 | < 0,001 |
| res_irregularity_mean | 0,9196 | 0,9162 | −0,033 | 0,007 |
| res_kurtosis_mean | 12,6 | 15,59 | +0,371 | < 0,001 |
| res_energy_std | 0,005963 | 0,003826 | −0,323 | < 0,001 |
| res_entropy_std | 0,5269 | 0,5024 | −0,151 | < 0,001 |
| res_prediction_error_std | 0,1429 | 0,1373 | −0,093 | < 0,001 |
| res_irregularity_std | 0,1004 | 0,08573 | −0,384 | < 0,001 |
| res_kurtosis_std | 15,27 | 15,23 | +0,019 | 0,125 |
| res_energy_median | 0,000931 | 0,0006287 | −0,253 | < 0,001 |
| res_entropy_median | 4,498 | 4,354 | −0,344 | < 0,001 |
| res_prediction_error_median | 0,007002 | 0,003742 | −0,362 | < 0,001 |
| res_irregularity_median | 0,9457 | 0,9356 | −0,154 | < 0,001 |
| res_kurtosis_median | 7,63 | 9,966 | +0,398 | < 0,001 |
| mod_band_0p5_2 | 0,2139 | 0,1929 | −0,097 | < 0,001 |
| mod_band_2_4 | 0,2726 | 0,2698 | −0,010 | 0,417 |
| mod_band_4_8 | 0,2891 | 0,3076 | +0,064 | < 0,001 |
| mod_band_8_20 | 0,1073 | 0,1206 | +0,139 | < 0,001 |
| mod_centroid_hz | 4,223 | 4,459 | +0,138 | < 0,001 |
| mod_entropy_nats | 2,631 | 2,768 | +0,191 | < 0,001 |
| mod_envelope_variability | 1,201 | 1,194 | −0,043 | < 0,001 |

Mann–Whitney U deskriptif pada LA train (2.580 REAL, 22.800 FAKE); utterance dalam satu speaker tidak independen, sehingga q bersifat deskriptif. Rank-biserial + berarti FAKE lebih tinggi.

## Tabel 4.3 Hasil LA eval (n = 17.275; 1.675 REAL)

| Keluarga | Classifier | AUC | EER (%) | BA (%) | F1 FAKE (%) | FPR (%) | FNR (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MFCC | LR | 0,707 | 35,8 | 64,7 | 68,3 | 23,8 | 46,8 |
| MFCC | HGB | 0,729 ± 0,001 | 34,2 | 67,4 | 71,6 | 22,3 | 42,9 |
| MFCC | GMM | 0,773 ± 0,006 | 30,2 | 70,5 | 78,2 | 25,0 | 34,1 |
| LFCC | LR | 0,839 | 24,3 | 75,3 | 71,5 | 5,3 | 44,1 |
| LFCC | HGB | 0,791 | 28,7 | 73,3 | 69,4 | 6,8 | 46,5 |
| LFCC | GMM | 0,860 ± 0,003 | 22,1 | 77,5 | 75,3 | 5,9 | 39,2 |
| MFCC_LFCC | LR | 0,862 | 22,1 | 78,1 | 73,9 | 2,6 | 41,2 |
| MFCC_LFCC | HGB | 0,821 | 25,5 | 76,2 | 72,9 | 5,2 | 42,3 |
| Residual | LR | 0,646 | 39,9 | 60,8 | 64,0 | 26,9 | 51,6 |
| Residual | HGB | 0,688 ± 0,011 | 36,2 | 62,0 | 58,6 | 18,3 | 57,7 |
| Modulation | LR | 0,638 | 41,0 | 59,0 | 71,6 | 40,2 | 41,9 |
| Modulation | HGB | 0,631 | 42,0 | 58,1 | 71,1 | 41,3 | 42,4 |
| Residual_Modulation | LR | 0,729 | 33,5 | 67,6 | 72,7 | 23,3 | 41,4 |
| Residual_Modulation | HGB | 0,754 ± 0,004 | 31,2 | 68,3 | 71,1 | 19,7 | 43,7 |
| QC_only | LR | 0,875 | 18,9 | 76,1 | 90,6 | 33,6 | 14,2 |
| QC_only | HGB | 0,876 | 18,2 | 78,2 | 90,1 | 28,0 | 15,5 |
| QC_plus_Residual_Modulation | LR | 0,867 | 20,4 | 80,3 | 85,9 | 15,9 | 23,4 |
| QC_plus_Residual_Modulation | HGB | 0,894 ± 0,001 | 18,1 | 82,6 | 85,8 | 10,8 | 24,1 |
| QC_plus_MFCC_LFCC | LR | 0,888 | 19,7 | 79,8 | 75,9 | 1,7 | 38,7 |
| QC_plus_MFCC_LFCC | HGB | 0,835 ± 0,001 | 24,4 | 77,1 | 73,8 | 4,6 | 41,2 |

Rerata ± SD tiga seed (SD ditampilkan bila > 0,0005; LR dan SVM deterministik). Threshold EER dari dev LA, dibekukan. GMM 64 komponen tingkat frame. BA = balanced accuracy.

## Tabel 4.4 Hasil DF (n = 812; 187 REAL)

| Keluarga | Classifier | AUC | EER (%) | BA (%) | F1 FAKE (%) | FPR (%) | FNR (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MFCC | LR | 0,732 | 33,1 | 66,0 | 57,6 | 9,6 | 58,4 |
| MFCC | HGB | 0,749 ± 0,004 | 31,4 | 67,8 | 61,9 | 10,7 | 53,7 |
| MFCC | GMM | 0,776 ± 0,004 | 30,4 | 72,0 | 72,9 | 16,2 | 39,8 |
| LFCC | LR | 0,701 | 36,3 | 64,5 | 68,8 | 27,8 | 43,2 |
| LFCC | HGB | 0,705 ± 0,002 | 36,5 | 64,1 | 68,3 | 28,0 | 43,8 |
| LFCC | GMM | 0,655 ± 0,028 | 38,1 | 65,5 | 82,6 | 49,9 | 19,1 |
| MFCC_LFCC | LR | 0,709 | 33,7 | 65,4 | 71,2 | 29,4 | 39,8 |
| MFCC_LFCC | HGB | 0,763 ± 0,001 | 32,1 | 68,5 | 64,6 | 12,5 | 50,5 |
| Residual | LR | 0,679 | 36,4 | 54,8 | 83,1 | 78,1 | 12,3 |
| Residual | HGB | 0,636 ± 0,006 | 41,4 | 57,5 | 78,9 | 62,4 | 22,7 |
| Modulation | LR | 0,502 | 51,3 | 48,7 | 63,6 | 57,2 | 45,4 |
| Modulation | HGB | 0,529 ± 0,002 | 47,6 | 52,1 | 61,0 | 45,8 | 50,1 |
| Residual_Modulation | LR | 0,673 | 37,4 | 55,0 | 81,6 | 74,3 | 15,7 |
| Residual_Modulation | HGB | 0,663 ± 0,010 | 39,4 | 57,7 | 79,6 | 63,3 | 21,4 |
| QC_only | LR | 0,686 | 35,3 | 62,5 | 89,1 | 72,7 | 2,2 |
| QC_only | HGB | 0,725 ± 0,002 | 33,6 | 64,5 | 89,4 | 68,4 | 2,6 |
| QC_plus_Residual_Modulation | LR | 0,765 | 33,6 | 60,6 | 87,7 | 74,3 | 4,5 |
| QC_plus_Residual_Modulation | HGB | 0,773 ± 0,009 | 31,0 | 63,2 | 87,3 | 66,5 | 7,1 |
| QC_plus_MFCC_LFCC | LR | 0,718 | 33,1 | 65,9 | 72,1 | 29,4 | 38,7 |
| QC_plus_MFCC_LFCC | HGB | 0,774 ± 0,003 | 31,7 | 68,9 | 65,3 | 12,5 | 49,7 |

Rerata ± SD tiga seed (SD ditampilkan bila > 0,0005; LR dan SVM deterministik). Threshold EER dari dev LA, dibekukan. GMM 64 komponen tingkat frame. BA = balanced accuracy.

## Tabel 4.5 Hasil MLAAD (n = 1.906; 953 REAL)

| Keluarga | Classifier | AUC | EER (%) | BA (%) | F1 FAKE (%) | FPR (%) | FNR (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MFCC | LR | 0,665 | 37,5 | 56,7 | 25,2 | 1,3 | 85,4 |
| MFCC | HGB | 0,775 ± 0,004 | 29,2 | 57,8 | 27,2 | 0,1 | 84,3 |
| MFCC | GMM | 0,678 ± 0,056 | 37,7 | 62,7 | 45,3 | 5,7 | 68,8 |
| LFCC | LR | 0,238 | 72,7 | 22,9 | 33,1 | 92,2 | 61,9 |
| LFCC | HGB | 0,174 ± 0,002 | 75,5 | 21,8 | 34,2 | 97,1 | 59,3 |
| LFCC | GMM | 0,253 ± 0,008 | 68,7 | 27,5 | 42,0 | 97,5 | 47,4 |
| MFCC_LFCC | LR | 0,193 | 77,8 | 24,3 | 38,0 | 97,8 | 53,6 |
| MFCC_LFCC | HGB | 0,164 ± 0,001 | 78,2 | 18,7 | 29,2 | 96,2 | 66,5 |
| Residual | LR | 0,466 | 51,7 | 43,7 | 54,7 | 80,5 | 32,1 |
| Residual | HGB | 0,427 ± 0,014 | 55,4 | 43,5 | 50,6 | 70,9 | 42,1 |
| Modulation | LR | 0,528 | 48,7 | 49,2 | 62,2 | 85,0 | 16,6 |
| Modulation | HGB | 0,487 ± 0,004 | 50,4 | 49,6 | 61,5 | 81,3 | 19,4 |
| Residual_Modulation | LR | 0,506 | 48,9 | 48,8 | 61,3 | 83,3 | 19,0 |
| Residual_Modulation | HGB | 0,513 ± 0,009 | 48,8 | 50,5 | 63,7 | 85,8 | 13,2 |
| QC_only | LR | 0,644 | 41,4 | 62,0 | 45,4 | 7,7 | 68,4 |
| QC_only | HGB | 0,421 ± 0,014 | 53,6 | 48,9 | 65,6 | 99,7 | 2,5 |
| QC_plus_Residual_Modulation | LR | 0,633 | 42,4 | 55,6 | 20,8 | 0,4 | 88,4 |
| QC_plus_Residual_Modulation | HGB | 0,525 ± 0,011 | 50,6 | 51,7 | 65,9 | 90,1 | 6,5 |
| QC_plus_MFCC_LFCC | LR | 0,198 | 73,1 | 47,2 | 64,1 | 100,0 | 5,6 |
| QC_plus_MFCC_LFCC | HGB | 0,166 ± 0,002 | 77,9 | 20,1 | 31,8 | 97,2 | 62,6 |

Rerata ± SD tiga seed (SD ditampilkan bila > 0,0005; LR dan SVM deterministik). Threshold EER dari dev LA, dibekukan. GMM 64 komponen tingkat frame. BA = balanced accuracy.

## Tabel 4.6 Selisih AUC berpasangan (seed 2026)

| Data | Perbandingan (A − B) | ΔAUC [CI 95%] | Speaker |
| --- | --- | --- | --- |
| LA eval | MFCC_LFCC__svm_rbf vs Residual_Modulation__rf | +0,155 [+0,125; +0,182] | 67 |
| LA eval | MFCC_LFCC__svm_rbf vs MFCC_LFCC__lr | −0,011 [−0,022; −0,001] | 67 |
| LA eval | MFCC_LFCC__svm_rbf vs Residual_Modulation__lr | +0,123 [+0,100; +0,145] | 67 |
| LA eval | Residual_Modulation__rf vs MFCC_LFCC__lr | −0,166 [−0,192; −0,137] | 67 |
| LA eval | Residual_Modulation__rf vs Residual_Modulation__lr | −0,031 [−0,051; −0,013] | 67 |
| LA eval | MFCC_LFCC__lr vs Residual_Modulation__lr | +0,135 [+0,111; +0,157] | 67 |
| LA eval | QC_plus_Residual_Modulation vs QC_only (lr) | −0,008 [−0,019; +0,003] | 67 |
| LA eval | QC_plus_Residual_Modulation vs QC_only (hgb) | +0,017 [+0,006; +0,029] | 67 |
| LA eval | QC_plus_MFCC_LFCC vs QC_only (lr) | +0,013 [−0,004; +0,029] | 67 |
| LA eval | QC_plus_MFCC_LFCC vs QC_only (hgb) | −0,040 [−0,058; −0,025] | 67 |
| LA eval | Residual_Modulation vs MFCC_LFCC (lr) | −0,135 [−0,157; −0,111] | 67 |
| LA eval | Residual_Modulation vs MFCC_LFCC (hgb) | −0,065 [−0,087; −0,042] | 67 |
| DF | MFCC_LFCC__svm_rbf vs Residual_Modulation__rf | +0,069 [+0,004; +0,153] | 72 |
| DF | MFCC_LFCC__svm_rbf vs MFCC_LFCC__lr | −0,016 [−0,047; +0,018] | 72 |
| DF | MFCC_LFCC__svm_rbf vs Residual_Modulation__lr | +0,024 [−0,038; +0,096] | 72 |
| DF | Residual_Modulation__rf vs MFCC_LFCC__lr | −0,085 [−0,172; −0,014] | 72 |
| DF | Residual_Modulation__rf vs Residual_Modulation__lr | −0,046 [−0,095; +0,003] | 72 |
| DF | MFCC_LFCC__lr vs Residual_Modulation__lr | +0,039 [−0,028; +0,115] | 72 |
| DF | QC_plus_Residual_Modulation vs QC_only (lr) | +0,077 [+0,009; +0,142] | 72 |
| DF | QC_plus_Residual_Modulation vs QC_only (hgb) | +0,049 [−0,018; +0,105] | 72 |
| DF | QC_plus_MFCC_LFCC vs QC_only (lr) | +0,030 [−0,058; +0,130] | 72 |
| DF | QC_plus_MFCC_LFCC vs QC_only (hgb) | +0,050 [−0,030; +0,130] | 72 |
| DF | Residual_Modulation vs MFCC_LFCC (lr) | −0,039 [−0,115; +0,028] | 72 |
| DF | Residual_Modulation vs MFCC_LFCC (hgb) | −0,091 [−0,168; −0,032] | 72 |

Bootstrap persentil berklaster speaker, 1.000 draw, utterance identik. Pasangan tanpa label config adalah pasangan bawaan antara champion dev (MFCC_LFCC SVM-RBF, Residual_Modulation RF) dan classifier acuan LR. MLAAD tidak didukung (3 speaker).

## Tabel 4.7 AUC per serangan, LA eval (sistem champion dev)

| Serangan | MFCC_LFCC SVM | Residual_Modulation RF | Selisih (RM − baseline) |
| --- | --- | --- | --- |
| A07 | 1,000 | 0,635 | −0,365 |
| A08 | 1,000 | 0,903 | −0,097 |
| A09 | 1,000 | 0,630 | −0,370 |
| A10 | 0,601 | 0,632 | +0,031 |
| A11 | 0,984 | 0,666 | −0,318 |
| A12 | 0,320 | 0,649 | +0,329 |
| A13 | 0,557 | 0,827 | +0,271 |
| A14 | 0,912 | 0,763 | −0,149 |
| A15 | 0,944 | 0,707 | −0,237 |
| A16 | 0,999 | 0,708 | −0,291 |
| A17 | 0,879 | 0,565 | −0,313 |
| A18 | 0,869 | 0,624 | −0,245 |
| A19 | 1,000 | 0,733 | −0,267 |

Champion dipilih dari AUC dev: MFCC_LFCC SVM-RBF dan Residual_Modulation RF. Rerata tiga seed; tiap serangan dibandingkan dengan seluruh REAL eval.

## Tabel 4.8 Kontrol jalan pintas hening: AUC gabungan vs di dalam strata

| Data | Sistem | AUC gabungan | AUC dalam desil hening | AUC dalam desil durasi |
| --- | --- | --- | --- | --- |
| LA eval | MFCC_LFCC__hgb | 0,821 | 0,849 | 0,840 |
| LA eval | MFCC_LFCC__lr | 0,862 | 0,899 | 0,868 |
| LA eval | Modulation__hgb | 0,631 | 0,595 | 0,598 |
| LA eval | Modulation__lr | 0,638 | 0,575 | 0,601 |
| LA eval | QC_only__hgb | 0,876 | 0,684 | 0,857 |
| LA eval | QC_only__lr | 0,875 | 0,644 | 0,883 |
| LA eval | Residual_Modulation__hgb | 0,754 | 0,713 | 0,742 |
| LA eval | Residual_Modulation__lr | 0,729 | 0,648 | 0,707 |
| LA eval | Residual__hgb | 0,688 | 0,675 | 0,706 |
| LA eval | Residual__lr | 0,646 | 0,578 | 0,629 |
| DF | MFCC_LFCC__hgb | 0,763 | 0,777 | 0,765 |
| DF | MFCC_LFCC__lr | 0,709 | 0,706 | 0,706 |
| DF | Modulation__hgb | 0,529 | 0,511 | 0,532 |
| DF | Modulation__lr | 0,502 | 0,508 | 0,512 |
| DF | QC_only__hgb | 0,725 | 0,544 | 0,734 |
| DF | QC_only__lr | 0,686 | 0,491 | 0,735 |
| DF | Residual_Modulation__hgb | 0,663 | 0,649 | 0,679 |
| DF | Residual_Modulation__lr | 0,673 | 0,658 | 0,708 |
| DF | Residual__hgb | 0,636 | 0,634 | 0,641 |
| DF | Residual__lr | 0,679 | 0,660 | 0,692 |

AUC durasi hening sendiri (diorientasi): asvspoof2019_la/dev 0,740; asvspoof2019_la/eval 0,890; asvspoof2021_df/external 0,763; mlaad_tiny/external 0,899. REAL memiliki hening tepi lebih panjang. AUC dalam desil = rerata berbobot AUC di 10 bin kuantil; deskriptif, bin dari data target.

## Tabel 4.9 Grouped permutation importance, Residual_Modulation LR seed 2026

| Data | Kelompok diacak | Penurunan AUC (rerata ± SD) |
| --- | --- | --- |
| asvspoof2019_la | Residual | 0,149 ± 0,007 |
| asvspoof2019_la | Modulation | 0,087 ± 0,006 |
| asvspoof2021_df | Residual | 0,171 ± 0,021 |
| asvspoof2021_df | Modulation | 0,006 ± 0,014 |
| mlaad_tiny | Residual | −0,038 ± 0,009 |
| mlaad_tiny | Modulation | 0,038 ± 0,009 |

Ketergantungan prediksi, bukan bukti kausal.

## Tabel 4.10 Konsistensi keputusan champion baseline (MFCC_LFCC SVM) dan bukti (RM RF)

| Data | Spearman ρ | Agreement (%) | Kappa | Keduanya benar (%) | Hanya baseline benar (%) | Hanya RM benar (%) | Keduanya salah (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LA eval | 0,111 | 51,9 | 0,044 | 26,8 | 33,4 | 14,7 | 25,1 |
| DF | 0,102 | 56,8 | 0,074 | 43,1 | 27,8 | 15,4 | 13,7 |
| MLAAD | 0,085 | 61,7 | 0,055 | 14,2 | 6,3 | 31,9 | 47,6 |

Rerata tiga seed pada threshold beku.

## Tabel 4.11 ΔAUC kondisi uji terhadap referensi, LA eval (subset n = 691; 67 REAL), LR

| Varian | MFCC_LFCC | Residual_Modulation | Residual | QC_only |
| --- | --- | --- | --- | --- |
| AUC referensi | 0,873 | 0,781 | 0,717 | 0,883 |
| LPC orde 12 | +0,000 [+0,000; +0,000] | +0,006 [−0,014; +0,027] | −0,000 [−0,037; +0,039] | +0,000 [+0,000; +0,000] |
| LPC orde 20 | +0,000 [+0,000; +0,000] | −0,006 [−0,020; +0,006] | −0,006 [−0,032; +0,015] | +0,000 [+0,000; +0,000] |
| Trim 25 dB | −0,005 [−0,010; +0,000] | −0,018 [−0,045; +0,008] | −0,052 [−0,086; −0,024] | −0,004 [−0,011; +0,003] |
| Trim 45 dB | +0,002 [−0,002; +0,005] | −0,053 [−0,081; −0,026] | −0,087 [−0,127; −0,048] | −0,013 [−0,040; +0,010] |
| Tanpa trim | +0,002 [−0,002; +0,006] | −0,140 [−0,180; −0,101] | −0,195 [−0,238; −0,146] | −0,089 [−0,127; −0,053] |
| Noise SNR 30 dB | −0,133 [−0,171; −0,096] | −0,074 [−0,123; −0,024] | −0,089 [−0,158; −0,023] | +0,003 [−0,002; +0,009] |
| Noise SNR 20 dB* | +0,004 [−0,033; +0,042] | +0,073 [+0,029; +0,121] | +0,165 [+0,122; +0,215] | −0,485 [−0,541; −0,426] |
| Noise SNR 10 dB* | −0,052 [−0,100; −0,005] | +0,034 [−0,019; +0,086] | +0,162 [+0,119; +0,213] | −0,427 [−0,484; −0,364] |
| MP3 128 kbps | −0,203 [−0,250; −0,161] | −0,047 [−0,071; −0,025] | −0,078 [−0,114; −0,039] | −0,000 [−0,002; +0,002] |
| MP3 64 kbps | −0,215 [−0,265; −0,163] | −0,060 [−0,090; −0,033] | −0,097 [−0,140; −0,058] | +0,002 [−0,002; +0,005] |
| MP3 32 kbps | −0,286 [−0,348; −0,224] | −0,112 [−0,151; −0,074] | −0,163 [−0,218; −0,105] | −0,008 [−0,030; +0,008] |

Model seed 2026 dan threshold dev dibekukan; subset tetap, 1 utterance per sel speaker × serangan. CI 95% bootstrap berpasangan berklaster speaker. *SNR 20/10 dB: noise ditambahkan sebelum trim, sehingga trim 35 dB tidak lagi memangkas hening (durasi hening terpangkas jadi 0); hasilnya interaksi noise × trim, bukan uji ketahanan noise murni.

## Tabel 4.12 ΔAUC kondisi uji terhadap referensi, DF (subset n = 416; 62 REAL), LR

| Varian | MFCC_LFCC | Residual_Modulation | Residual | QC_only |
| --- | --- | --- | --- | --- |
| AUC referensi | 0,769 | 0,649 | 0,645 | 0,758 |
| LPC orde 12 | +0,000 [+0,000; +0,000] | −0,024 [−0,048; +0,001] | −0,024 [−0,054; +0,010] | +0,000 [+0,000; +0,000] |
| LPC orde 20 | +0,000 [+0,000; +0,000] | −0,012 [−0,030; +0,004] | −0,022 [−0,042; −0,003] | +0,000 [+0,000; +0,000] |
| Trim 25 dB | +0,002 [−0,003; +0,007] | −0,006 [−0,034; +0,019] | −0,017 [−0,045; +0,006] | +0,014 [−0,000; +0,032] |
| Trim 45 dB | −0,000 [−0,003; +0,002] | −0,031 [−0,061; −0,005] | −0,025 [−0,054; −0,000] | −0,053 [−0,088; −0,027] |
| Tanpa trim | −0,000 [−0,003; +0,002] | −0,085 [−0,138; −0,034] | −0,080 [−0,132; −0,033] | −0,215 [−0,292; −0,139] |
| Noise SNR 30 dB | −0,118 [−0,193; −0,039] | −0,050 [−0,118; +0,021] | −0,045 [−0,130; +0,041] | +0,003 [−0,004; +0,009] |
| Noise SNR 20 dB* | −0,014 [−0,098; +0,076] | +0,049 [−0,026; +0,132] | +0,123 [+0,029; +0,232] | −0,460 [−0,564; −0,358] |
| Noise SNR 10 dB* | −0,053 [−0,144; +0,046] | +0,007 [−0,071; +0,095] | +0,107 [+0,010; +0,216] | −0,447 [−0,546; −0,348] |
| MP3 128 kbps | −0,169 [−0,257; −0,084] | −0,025 [−0,052; +0,004] | −0,038 [−0,071; −0,003] | −0,015 [−0,048; +0,006] |
| MP3 64 kbps | −0,159 [−0,240; −0,068] | −0,023 [−0,051; +0,008] | −0,028 [−0,064; +0,008] | −0,019 [−0,058; +0,007] |
| MP3 32 kbps | −0,167 [−0,260; −0,074] | −0,070 [−0,110; −0,024] | −0,051 [−0,102; +0,005] | −0,028 [−0,075; +0,009] |

Model seed 2026 dan threshold dev dibekukan; subset tetap, 1 utterance per sel speaker × serangan. CI 95% bootstrap berpasangan berklaster speaker. *SNR 20/10 dB: noise ditambahkan sebelum trim, sehingga trim 35 dB tidak lagi memangkas hening (durasi hening terpangkas jadi 0); hasilnya interaksi noise × trim, bukan uji ketahanan noise murni.

## Tabel 4.13 Balanced accuracy pada threshold beku, LA eval subset (LR, %)

| Varian | MFCC_LFCC | QC_only | Residual_Modulation |
| --- | --- | --- | --- |
| Referensi | 78,3 | 76,3 | 70,1 |
| LPC orde 12 | 78,3 | 76,3 | 67,2 |
| LPC orde 20 | 78,3 | 76,3 | 72,6 |
| Trim 25 dB | 78,6 | 79,1 | 70,2 |
| Trim 45 dB | 78,1 | 73,8 | 67,4 |
| Tanpa trim | 77,8 | 58,3 | 62,6 |
| Noise SNR 30 dB | 57,4 | 77,0 | 61,9 |
| Noise SNR 20 dB* | 50,1 | 49,8 | 62,3 |
| Noise SNR 10 dB* | 50,0 | 50,0 | 50,1 |
| MP3 128 kbps | 50,0 | 81,3 | 63,7 |
| MP3 64 kbps | 50,0 | 81,6 | 64,1 |
| MP3 32 kbps | 53,2 | 82,1 | 58,2 |

## Tabel 4.14 AUC ablation fitur modulasi (rerata tiga seed)

| Model | LA eval LR | LA eval HGB | DF LR | DF HGB | MLAAD LR | MLAAD HGB |
| --- | --- | --- | --- | --- | --- | --- |
| Modulasi lengkap (7) | 0,638 | 0,631 | 0,502 | 0,529 | 0,528 | 0,487 |
| Bentuk saja (centroid, entropy, variabilitas) | 0,641 | 0,634 | 0,497 | 0,529 | 0,529 | 0,497 |
| Empat pita saja | 0,543 | 0,533 | 0,531 | 0,536 | 0,543 | 0,510 |
| Pita 0,5–2 Hz saja | 0,551 | 0,543 | 0,544 | 0,528 | 0,535 | 0,527 |
| Pita 2–4 Hz saja | 0,486 | 0,504 | 0,465 | 0,477 | 0,570 | 0,491 |
| Pita 4–8 Hz saja | 0,545 | 0,539 | 0,490 | 0,498 | 0,555 | 0,556 |
| Pita 8–20 Hz saja | 0,520 | 0,518 | 0,523 | 0,521 | 0,485 | 0,482 |
| Residual + modulasi | 0,729 | 0,754 | 0,673 | 0,663 | 0,506 | 0,513 |
| MFCC + LFCC | 0,862 | 0,821 | 0,709 | 0,763 | 0,193 | 0,164 |

Energi pita adalah proporsi yang berjumlah ≈ 1 (median 0,992), sehingga desain "tanpa satu pita" tidak dapat diidentifikasi; ablation memakai tiap pita sendirian. Model dilatih ulang dengan resep sama; fitur dari cache yang sama dengan run utama.

## Tabel 4.15 Selisih AUC berpasangan ablation modulasi

| Data | Perbandingan (A − B) | ΔAUC [CI 95%] |
| --- | --- | --- |
| LA eval | Modulation vs Modulation_bands_only (hgb) | +0,100 [+0,084; +0,114] |
| LA eval | Modulation vs Modulation_bands_only (lr) | +0,095 [+0,081; +0,110] |
| LA eval | Modulation vs Modulation_shape_only (hgb) | −0,002 [−0,008; +0,004] |
| LA eval | Modulation vs Modulation_shape_only (lr) | −0,003 [−0,006; −0,001] |
| LA eval | Modulation_bands_only vs Modulation_band_0p5_2_only (hgb) | −0,010 [−0,026; +0,008] |
| LA eval | Modulation_bands_only vs Modulation_band_0p5_2_only (lr) | −0,009 [−0,024; +0,006] |
| LA eval | Modulation_bands_only vs Modulation_band_2_4_only (hgb) | +0,030 [+0,007; +0,051] |
| LA eval | Modulation_bands_only vs Modulation_band_2_4_only (lr) | +0,057 [+0,032; +0,083] |
| LA eval | Modulation_bands_only vs Modulation_band_4_8_only (hgb) | −0,005 [−0,023; +0,011] |
| LA eval | Modulation_bands_only vs Modulation_band_4_8_only (lr) | −0,002 [−0,020; +0,015] |
| LA eval | Modulation_bands_only vs Modulation_band_8_20_only (hgb) | +0,015 [+0,003; +0,028] |
| LA eval | Modulation_bands_only vs Modulation_band_8_20_only (lr) | +0,023 [+0,012; +0,034] |
| DF | Modulation vs Modulation_bands_only (hgb) | −0,007 [−0,067; +0,048] |
| DF | Modulation vs Modulation_bands_only (lr) | −0,029 [−0,093; +0,030] |
| DF | Modulation vs Modulation_shape_only (hgb) | +0,003 [−0,015; +0,022] |
| DF | Modulation vs Modulation_shape_only (lr) | +0,006 [−0,002; +0,015] |
| DF | Modulation_bands_only vs Modulation_band_0p5_2_only (hgb) | +0,014 [−0,045; +0,074] |
| DF | Modulation_bands_only vs Modulation_band_0p5_2_only (lr) | −0,013 [−0,049; +0,027] |
| DF | Modulation_bands_only vs Modulation_band_2_4_only (hgb) | +0,064 [+0,000; +0,137] |
| DF | Modulation_bands_only vs Modulation_band_2_4_only (lr) | +0,067 [+0,005; +0,132] |
| DF | Modulation_bands_only vs Modulation_band_4_8_only (hgb) | +0,040 [−0,017; +0,102] |
| DF | Modulation_bands_only vs Modulation_band_4_8_only (lr) | +0,041 [−0,004; +0,091] |
| DF | Modulation_bands_only vs Modulation_band_8_20_only (hgb) | +0,018 [−0,015; +0,043] |
| DF | Modulation_bands_only vs Modulation_band_8_20_only (lr) | +0,007 [−0,020; +0,032] |

Median tiga seed dari batas bootstrap berklaster speaker (1.000 draw).
