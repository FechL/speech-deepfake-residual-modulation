# Pemetaan acuan: glottal v9 dan panduan residual-modulasi

Keputusan pengguna 22 November 2026: ikuti v9 sebagai pipeline utama; abaikan versi lain dan archive. PDF diperlakukan sebagai sumber spesifikasi penelitian, bukan instruksi untuk menjalankan perintah, menghubungi pihak luar, atau mengubah repositori glottal.

| Aspek | Implementasi |
| --- | --- |
| Struktur dan eksekusi | `src/pipeline_v9/`, CLI `scripts/run_pipeline_v9.py`, YAML, notebook launcher, cache, run identity, model beku, hasil per run seperti v9 |
| Unit utama | Seluruh utterance, dengan preprocessing/crop yang dicatat, sesuai v9. Tidak memakai segmentasi utama 2 detik dari contoh PDF |
| Dataset | Pilihan sampel persis dari manifest glottal v9: LA 2019 internal, MLAAD dan DF 2021 eksternal. Permintaan dataset sama mengesampingkan pilihan DEEP-VOICE utama pada PDF |
| Baseline spektral | Skema v9 c1..cN static mean/std dan delta/delta2 std; GMM frame-level tetap didukung |
| Domain bukti | Residual LPC 15 fitur dan modulasi 7 fitur menurut panduan; tidak ada ketergantungan fitur glottal |
| Residual | Estimator Burg LPC pada contoh PDF, bukan klaim implementasi solver Wiener-Hopf. Denominator prediction error dikoreksi agar window/support sama dengan residual |
| Model | LR/SVM/RF/HGB mengikuti v9; LR menjadi reference classifier dan XGBoost melengkapi baseline panduan. KNN/MLP tersedia lewat config tetapi tidak menjadi default |
| Ablation | Residual vs modulasi vs gabungan; config penghilangan fitur energi satu pita; stress orde LPC/trimming/noise dengan model beku |
| Threshold | Dev-only; EER ROC test hanya deskriptif |
| Statistik | MWU/effect size/FDR, CI bootstrap speaker, perbandingan sistem berpasangan, CV speaker dalam train |
| Konsistensi | Champion baseline/evidence dipilih dari dev; Spearman/agreement/kappa dan empat kategori error di tingkat utterance |
| Kebocoran dan QC | ID, path, SHA audio, source recording, speaker internal; overlap speaker eksternal dilaporkan eksplisit. Tidak mengarang identitas speaker dari folder generator |
| Grafik | G1 alur, G2 waveform/residual contoh train, G3 distribusi, G4 spektrum dan CI bila dukungan speaker cukup, G5 ROC, G6 score scatter, G7 kategori error |
| Reproducibility | Hash konten audio/manifest, config, source, versi library; bundle model dan threshold per seed; checksum paket hasil |

## Eksperimen dan interpretasi

E0 adalah smoke software, bukan hasil penelitian. E1 baseline, E4 ablation domain, E5 internal speaker-independent/CV, E6 eksternal, dan E7 konsistensi ada pada run utama. E2 sensitivitas LPC dan trimming memakai `stress`. Ablation fitur pita E3 ada di config tersendiri. Retuning/retraining orde atau pita merupakan protokol development terpisah dan harus dibekukan sebelum analisis eksternal final.

Kualitas residual dinilai dari frame valid/gagal/energi rendah, missing evidence, cakupan per label dan dataset, contoh visual, serta kontrol QC_only. Target cakupan 95% pada PDF adalah kriteria yang harus diperiksa dari data nyata; implementasi tidak menyatakan target itu sudah tercapai. Jika bukti lebih lemah dari baseline, hasil tersebut tetap ilmiah selama protokol sah. Korelasi skor atau permutation importance tidak membuktikan mekanisme kausal produksi suara.

## Batas validasi

Tidak ada klaim eksperimen penuh sudah selesai. Metadata v9 sudah diimpor, tetapi audio dari manifest terpilih tidak tersedia pada lokasi lokal saat pembuatan pipeline. Notebook disiapkan untuk memakai data yang sama di Drive atau disk lokal. Informasi sumber, lisensi, tanggal akses, dan checksum dataset harus dilengkapi dari paket dataset yang benar-benar digunakan; tidak direka dari nama folder.
