# Speech deepfake — residual dan modulasi, pipeline v9

Implementasi mandiri dengan **pipeline glottal v9 sebagai satu-satunya acuan struktur dan alur**. Versi v1–v8, kode arsip, hasil model glottal, dan pipeline pada repositori residual lama tidak dipakai. Panduan `docs/panduan.pdf` menjadi acuan definisi residual, modulasi, statistik, dan konsistensi bukti. Permintaan pengguna untuk mengikuti v9 menentukan ekstraksi seluruh utterance dan dataset.

Dua jalur dipertahankan terpisah:

1. **Klasifikasi:** MFCC, LFCC, MFCC+LFCC → LR/SVM/RF/HGB/XGBoost; MFCC/LFCC-GMM sebagai referensi tingkat frame seperti v9.
2. **Bukti:** residual LPC, modulasi, residual+modulasi → classifier yang sama, lalu perbandingan skor, keputusan, statistik dan kasus galat per utterance.

Residual adalah prediction error all-pole; bukan aliran glotal murni. Tidak ada fitur glottal, feature fusion dengan baseline, atau klaim likelihood ratio forensik. SVM memakai decision score; skor classifier lain tidak diklaim sebagai probabilitas forensik terkalibrasi. Skor lebih tinggi selalu berarti lebih deepfake (`real=0`, `fake=1`).

## Struktur

```text
configs/           konfigurasi utama dan ablation pita modulasi
src/pipeline_v9/   data, fitur, training, evaluasi, runtime, statistik, visualisasi, stress
scripts/           CLI dan generator notebook
notebooks/         launcher lokal/Colab v9 (kode inti tetap di src/)
  setup/           setup manifest dataset v9
manifests/         manifest independen proyek ini dan audit asal glottal v9
  glottal_v9/       pilihan sampel, label, split yang sama dengan run glottal v9
data/             raw/, processed/, external/, primary/, sample/, demo/
checkpoints/       model/imputer/scaler/fitur/threshold yang dibekukan per run dan seed
results/           cache fitur dan paket hasil per run
figures/           tempat ekspor gambar; gambar per run ada di results/<run>/figures/
exports/           ekspor hasil
docs/             panduan, AI-context/, AI-research/, pustaka/, proposal/
tests/             tes fitur, leakage, threshold, cache dan pipeline
```

Audio sumber dibaca dari folder dataset glottal/Drive; tidak disalin atau dimodifikasi. Seluruh turunan ditulis di proyek ini. Tidak perlu membuat arsip pipeline lama.

## Dataset dan manifest yang sama dengan glottal v9

Manifest awal sudah diimpor dari `speech-deepfake-glottal/results/forensic_whole_utterance_v9/fe17e2b856dd5d6c/manifests/`, mempertahankan **47.373** ID utterance, label, attack, dataset, dan split:

| Dataset | Train | Dev | Eval / eksternal | Peran |
| --- | ---: | ---: | ---: | --- |
| ASVspoof 2019 LA | 25.380 | 2.000 | 17.275 | Pengembangan dan uji internal |
| MLAAD-tiny | — | — | 1.906 | Eksternal |
| ASVspoof 2021 DF | — | — | 812 | Eksternal |

`ADDITIONAL` hanya masuk jika disertakan dalam manifest run v9 sumber; run ini tidak menyertakannya. Importer mendukungnya. Tidak ada penambahan sampel secara diam-diam.

**Kondisi lokal 22 November 2026:** 0/47.373 path audio terpilih tersedia. Metadata/protokol ada, tetapi eksperimen dataset penuh membutuhkan audio lokal atau Drive. Daftarnya di `manifests/glottal_v9/path_availability.csv`. Audit ini menguji keberadaan path, bukan checksum/decode audio. Ini bukan hasil penelitian atau kegagalan model.

Untuk lokasi lain, impor ulang manifest dengan `--data-root` yang sesuai. Importer memetakan path Colab lama; tidak menggunakan lokasi `/content/...` secara langsung:

```bash
python scripts/run_pipeline_v9.py import-glottal-v9 \
  --source-run ../speech-deepfake-glottal/results/forensic_whole_utterance_v9/fe17e2b856dd5d6c \
  --data-root ../speech-deepfake-glottal/data \
  --out manifests/glottal_v9
```

Untuk membangun LA penuh dari protokol resmi, gunakan `manifest-la --root <folder-LA> --out <manifest.csv>`. Ini menghasilkan populasi berbeda dari pilihan sampel v9 yang telah dibekukan; jangan menyebutnya perbandingan sampel identik.

## Menjalankan lokal

Python 3.11 disarankan dalam panduan; dependensi kompatibel dipasang melalui `requirements.txt`. Validasi implementasi di lingkungan tersedia menggunakan Python 3.14. Snapshot versi lingkungan pengujian ada di `requirements-lock.txt`; gunakan Python yang sesuai jika memakai snapshot tersebut.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tahap utama setelah audio tersedia:

```bash
# 1. Audit checksum, header, speaker/utterance/source/hash leakage.
#    (Alternatif: `stage` menyalin audio terpilih ke folder lokal dari zip/folder, lalu mengaudit.)
python scripts/run_pipeline_v9.py audit \
  --manifest manifests/glottal_v9/manifest.csv \
  --out manifests/glottal_v9/audited_manifest.csv

# 2. Ekstraksi seluruh utterance dalam shard tulis-sekali; dapat dilanjutkan dengan command yang sama.
#    Tambahkan --backup <folder> untuk mencadangkan setiap shard dan memulihkannya pada sesi baru.
python scripts/run_pipeline_v9.py extract \
  --manifest manifests/glottal_v9/audited_manifest.csv \
  --config configs/residual_modulation_v9.yaml \
  --out results/residual_modulation_v9/features --jobs 2

# 3. Train/dev, bekukan semua seed, baru evaluasi target.
python scripts/run_pipeline_v9.py run \
  --features results/residual_modulation_v9/features \
  --config configs/residual_modulation_v9.yaml

# 4. Sensitivitas pada subset target tetap, tanpa retraining.
python scripts/run_pipeline_v9.py stress \
  --run results/residual_modulation_v9/<RUN_ID> \
  --features results/residual_modulation_v9/features --per-cell 2 --jobs 2
```

`audit --allow-missing` hanya membuat audit ketika audio belum lengkap. Perintah `run` menolak kegagalan ekstraksi secara default. `--allow-failures` memberi label run `available_subset` dan mencatat jumlah kasus yang dikeluarkan; tidak mengubah sampel secara otomatis.

Uji perangkat lunak tanpa dataset:

```bash
python scripts/run_pipeline_v9.py smoke --out results/synthetic_smoke
```

Perintah ini membuat **40 sinyal sintetis**, tiga seed, lima classifier, dua GMM, dan laporan lengkap. Sinyal tersebut hanya menguji implementasi dan tidak boleh dimasukkan dalam hasil skripsi.

## Menjalankan Colab

Buka `notebooks/pipeline_forensic_whole_utterance_v9.ipynb`, atur path pada T00, lalu jalankan T00–T06 berurutan. Kode dibaca dari folder proyek di Drive dan audio dari folder dataset glottal. Notebook tidak berisi token atau kode inti yang diduplikasi.

Alurnya dirancang agar tahan putus dan tidak membebani Drive:

| Lokasi | Isi |
| --- | --- |
| `/content/work` (disk lokal runtime) | manifest, audio hasil staging, fitur, frame, model, dan hasil; semua proses berat |
| `backup/<PROFILE>/` di Drive | `inputs/` (manifest, config, audit), `features/<identitas>/shards/` (CSV + tar frame per ±500 utterance), `checkpoints/<RUN_ID>/` (seed beku), `runs/<RUN_ID>/` (paket hasil) |

- **T00b** mengunggah 1 MB uji ke Drive akun Colab lalu menghapusnya. Kuota penuh langsung terdeteksi sebelum bekerja berjam-jam. File yang ditulis Colab memakai kuota **akun yang login di Colab**, walau foldernya milik akun lain (shortcut).
- **T03 `stage`** menyalin hanya utterance terpilih ke disk lokal: LA dari `LA.zip` yang disalin lokal lebih dulu, DF langsung dari `DF.zip`, MLAAD per file. SHA-256 dihitung dari byte yang ditulis dan menjadi dasar identitas cache. Folder glottal hanya dibaca.
- **T04 `extract --backup`** menulis shard tulis-sekali. Tidak ada file besar yang ditulis ulang di Drive, sehingga tidak menumpuk revisi atau antrean upload. Sesi baru memulihkan shard dari cadangan dan hanya mengekstrak sisanya.
- **T05 `run --backup`** mencadangkan setiap seed yang sudah beku dan paket hasil akhir; sesi baru memulihkannya.
- Identitas cache fitur hanya bergantung pada byte audio, pengaturan fitur, modul penghitung fitur (`audio_io`, `channels`, `common`, `extraction`, `features`), dan versi library numerik. Path, lokasi folder kerja, atau perbaikan kode training/laporan tidak membatalkan cache.

Bila runtime terputus, jalankan ulang dari T00. Staging audio diulang (±15–25 menit), lalu T04/T05 melanjutkan dari cadangan. `PROFILE='full'` mempertahankan pilihan sampel v9. `PROFILE='smoke'` membuat subset eksplisit dan konfigurasi kecil; hasilnya tidak boleh dilabeli eksperimen penuh.

## Fitur dan kontrol v9

Semua jalur memakai urutan preprocessing v9: mono, crop tengah maksimum 60 s pada sample rate asli, resample 16 kHz, trimming tepi 35 dB, pengurangan DC, normalisasi RMS per utterance, dan minimum bersama 0,30 s. Frame aktif = RMS di atas persentil-90 RMS frame dikurangi 30 dB (aturan v9); statistik spektral membutuhkan minimal 10 frame aktif. `qc_rms` adalah RMS sebelum normalisasi. Audio yang sepenuhnya sunyi tetap menghasilkan baris dengan bukti NaN (tidak melempar galat seperti v9). Kesetaraan MFCC/LFCC dengan `extract_audio_v9` diperiksa pada sinyal uji (selisih 0). Set `max_seconds: null` bila seluruh durasi diperlukan; konfigurasi berubah menghasilkan identitas cache berbeda. Tidak ada gate voiced/pitch khusus bukti.

- **MFCC/LFCC:** c1–c20, static mean/std, delta std, delta2 std; frame 25 ms/hop 10 ms dan frame aktif, sesuai skema v9. Tidak mencampur statistik spektral contoh panduan dengan baseline v9.
- **Residual:** LPC Burg (`librosa.lpc`, contoh panduan), p=16; energi, entropy histogram (bit), rasio energi prediction error, irregularity, kurtosis Pearson. Ringkasan mean/std/median frame menjadi **15 fitur**. Frame Hamming; startup filter p sampel dibuang dan rasio energi memakai support serta window yang sama. Ini koreksi terhadap contoh panduan yang membandingkan denominator sebelum windowing. Frame rendah energi dan frame numerik gagal dihitung terpisah. Bukan solver Wiener-Hopf eksplisit.
- **Modulasi:** Hilbert envelope → lowpass 40 Hz → resample 100 Hz → mean removal → Hann → power FFT 0–20 Hz. **7 fitur:** proporsi energi 0,5–2 / 2–4 / 4–8 / 8–20 Hz, centroid (Hz), entropy (nat), dan `std(centered envelope)/mean(abs(centered envelope))`. Variability ini mengikuti formula panduan, bukan coefficient of variation envelope asli. Normalisasi pita memakai seluruh power (0,20] Hz; ujung 20 Hz disertakan. Envelope tanpa dinamika menghasilkan NaN, bukan bukti valid palsu.
- **QC_only:** durasi, peak asli, RMS sebelum normalisasi, rasio frame aktif, dan rasio residual valid. QC tidak otomatis dimasukkan ke jalur bukti. HGB menangani NaN native; missingness tetap dapat menjadi sinyal model. Missing indicators classifier lain default mati. Complete-case evidence dilaporkan sebagai populasi tambahan, bukan pengganti hasil utama.

Bandingkan keluarga dengan classifier yang sama melalui `reference_classifier: lr`, serta sistem champion per keluarga yang dipilih dari AUC dev. Threshold EER dipilih pada dev dan dibekukan; EER test merupakan statistik ROC deskriptif, bukan threshold operasi baru.

## Ablation dan statistik

Residual-only, modulation-only, dan residual+modulasi sudah ada di config utama. Config `residual_modulation_v9_band_ablation.yaml` menambahkan penghilangan satu fitur energi pita pada model modulasi. Centroid/entropy tetap mengandung informasi spektrum penuh: ini **ablation fitur pita**, bukan penghapusan informasi frekuensi total. Gunakan anggaran model yang sama dan pilih dari train/dev sebelum membaca hasil eksternal.

`stress` mengubah satu faktor orde LPC 12/20, trim 25/45 dB/no trim, noise SNR30/20/10, atau MP3 128/64/32 kbps pada subset target tetap. Model seed pertama dan threshold tidak dilatih ulang; GMM tidak disertakan. Untuk eksperimen orde LPC dengan retraining, buat config development terpisah sebelum pembekuan eksternal.

Uji fitur menggunakan Mann–Whitney, rank-biserial (positif = fake cenderung lebih tinggi), FDR BH, dan CI selisih median bootstrap speaker. Uji MWU bersifat deskriptif karena utterance satu speaker dapat berkorelasi. Tidak dipakai sebagai seleksi fitur. CV menggunakan speaker dalam train saja. CI performa AUC/EER/F1 dan selisih sistem memakai resampling speaker berpasangan, minimal 5 speaker teridentifikasi; dukungan lebih kecil dicatat `unsupported`, bukan diganti bootstrap segmen. SD antarseed bukan CI.

MLAAD source-speaker ID bukan bukti identitas suara keluaran sintesis. Cross-dataset tidak otomatis speaker-independent; overlap speaker dengan development dicatat. Audio/hash/source recording lintas split ditolak. Identitas speaker `unknown` tidak boleh dipakai untuk membentuk train/dev speaker-independent.

## Hasil

`results/residual_modulation_v9/<RUN_ID>/` berisi:

- `run_identity.json`, `manifest.csv`, `features.csv`, `measurement_coverage.csv`, `leakage_audit.json`.
- `development_selection.csv`, `grouped_cv.csv`, `utterance_scores.csv`, `metrics.csv`, `seed_summary.csv`.
- `statistical_tests.csv`, `bootstrap_intervals.csv`, `grouped_permutation_importance.csv`.
- `evidence_consistency.csv`, `error_analysis.csv` (empat kategori benar/salah).
- `figures/G1…G7*.png`, data spektrum G4 dan indeks contoh QC G2.
- `complete.json` dengan checksum paket hasil.

Model, imputer, scaler, daftar fitur, dan threshold ada di `checkpoints/residual_modulation_v9/<RUN_ID>/seed_<SEED>/`, bersama `freeze.json`. Cache fitur terikat checksum audio, pengaturan fitur, modul penghitung fitur, serta versi library numerik; perubahan identitas memerlukan output ekstraksi baru. Run training terikat identitas fitur, tabel fitur, manifest, seluruh kode, dan config. Perintah yang sama dapat melanjutkan cache yang identik.

Seperti v9, keputusan utama menggunakan fitur seluruh utterance, bukan agregasi segmen 2 detik. Karena itu tidak dibuat `segment_scores.csv` yang menyiratkan pipeline segmentasi panduan telah dijalankan. Penjelasan pemetaan lengkap ada di `docs/AI-context/pemetaan-v9-panduan.md`.
