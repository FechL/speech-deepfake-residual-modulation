# Implementasi pipeline residual dan modulasi v9

Tanggal: 22 November 2026, Asia/Jakarta.

## Hasil implementasi

Pipeline mandiri tersedia di `src/pipeline_v9/`, CLI `scripts/run_pipeline_v9.py`, konfigurasi YAML, dan notebook `notebooks/pipeline_forensic_whole_utterance_v9.ipynb`. Acuan alur hanya glottal v9 sesuai arahan pengguna. Tidak ada pemakaian kode v1–v8 atau arsip. Seluruh model dan hasil berada di proyek residual-modulasi.

Fitur bukti diganti menjadi 15 ringkasan residual LPC serta 7 fitur modulasi. Baseline MFCC/LFCC mengikuti skema seluruh utterance v9, bersama GMM tingkat frame. LR/SVM/RF/HGB dipertahankan dari v9; XGBoost ditambahkan untuk baseline panduan. HGB memakai NaN native. Missing indicators classifier lain default mati; missingness tetap perlu diaudit bersama cakupan, QC dan complete-case.

Alur meliputi manifest, audit data, ekstraksi resumable, fitting train-only, pemilihan champion/threshold dev-only, pembekuan semua seed sebelum evaluasi, statistik fitur, CV speaker, CI berpasangan, konsistensi, panel attack, permutation importance kelompok, grafik G1–G7, dan stress model beku. Model yang sama dibandingkan antar keluarga; champion sistem dipisahkan dari perbandingan classifier yang sama.

## Dataset

Impor dari run glottal v9 `fe17e2b856dd5d6c` menghasilkan 47.373 utterance. Verifikasi langsung menyatakan kolom `utterance_id`, `label`, `speaker_id`, `attack_id`, `dataset`, dan `split` identik terhadap manifest sumber setelah pengurutan. Path Colab dipetakan ke lokasi data lokal dan path asal dipertahankan pada `source_path`. Metadata lolos audit speaker internal. Audit checksum audio belum dapat dilakukan untuk populasi ini karena file terpilih tidak tersedia pada path lokal.

Komposisi: LA train 25.380, dev 2.000, eval 17.275; MLAAD eksternal 1.906; DF eksternal 812. ADDITIONAL tidak ada dalam run sumber ini dan tidak ditambahkan. Importer mendukung manifest ADDITIONAL v9 jika dipilih di run sumber yang berbeda.

Status audio lokal terpilih: **0/47.373 tersedia**. Daftar ada pada `manifests/glottal_v9/path_availability.csv`. Tidak ada penggantian file atau pelatihan pada subset tersedia tanpa penandaan. Untuk eksperimen penuh, jalankan notebook dengan data Drive yang sama atau arahkan importer ke staging audio yang sesuai.

## Validasi perangkat lunak

- `python -m pytest -q`: **14 passed**. Mencakup LPC/energy ratio, lokalisasi pita modulasi pada sinyal known-envelope, NaN silence, stereo/resampling/crop, speaker split/hash leakage, threshold beku, FDR, dukungan bootstrap, scaler train-only, kategori konsistensi, cache identity/tamper/rebuild, deterministic SNR, dan MP3 roundtrip.
- CLI `smoke`: **40 sinyal sintetis**, tiga seed, lima classifier untuk tujuh keluarga dan MFCC/LFCC-GMM. Train/dev/eval/external dijalankan sampai laporan selesai. Ada 54 artifact utama yang checksum-nya diverifikasi. External sintetis sengaja hanya dua speaker sehingga CI ditandai unsupported; tidak diganti bootstrap utterance.
- CLI `stress`: **12 kondisi** sukses pada target sintetis yang tetap: reference, LPC12/20, trim25/45/no-trim, noise30/20/10, MP3 128/64/32. Threshold/model tetap, interval berpasangan dan perubahan fitur tersimpan. GMM tidak masuk stress seperti v9.
- Pemanggilan ulang `run` memverifikasi artifact dan memakai hasil lengkap yang identik.
- Tiga cuplikan WAV nyata lokal dari ADDITIONAL (maksimum tiga detik, satu REAL dan dua FAKE) berhasil diekstraksi. Ini pemeriksaan format/audio, bukan eksperimen dataset atau estimasi performa.
- Dua notebook lolos validasi schema nbformat dan syntax Python semua sel. Sel Colab tidak dijalankan terhadap Drive dalam sesi lokal ini.
- Grafik G1, G4, dan G6 hasil sintetis diperiksa visual dan dapat dibaca.

Lingkungan pengujian memakai interpreter proyek glottal yang sudah tersedia (Python 3.14) untuk menghindari instalasi baru; kode tidak mengimpor modul dari proyek glottal. Snapshot dependensi ada di `requirements-lock.txt`. Environment baru dapat dibuat memakai `requirements-dev.txt`; kompatibilitas environment Python 3.11 belum diuji dalam sesi ini.

Lokasi bukti software: `results/synthetic_smoke/runs/b13424ffa9d3c33a/`, `results/validation/verification_summary.json`, dan `results/validation/real_audio_feature_check.csv`. Semua skor sintetis diberi dataset `SYNTHETIC_SMOKE` dan tidak boleh dipakai untuk hasil skripsi.

## Batas dan keputusan metode

Ekstraksi utama mengikuti v9 seluruh utterance, bukan segmentasi 2 detik pada contoh panduan. Skor segmen tidak dibuat. Residual memakai Burg LPC dari contoh PDF, bukan solver Wiener-Hopf eksplisit. Rasio prediction error memakai window/support yang sama; transient awal LPC dibuang. Ablation pita adalah penghilangan satu fitur energi pita, bukan penghapusan seluruh informasi frekuensi yang masih dapat direpresentasikan centroid/entropy.

CI berbasis speaker membutuhkan minimal lima speaker teridentifikasi. Statistik MWU/FDR bersifat deskriptif terhadap ketergantungan utterance dalam speaker. Korelasi, agreement dan permutation importance tidak membuktikan kausalitas atau validitas bukti hukum. Target cakupan residual 95% dan hasil penelitian nyata belum dapat dinilai sampai audio manifest tersedia.
