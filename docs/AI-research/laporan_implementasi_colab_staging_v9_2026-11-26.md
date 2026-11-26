# Implementasi eksekusi Colab, cadangan, dan kontrol jalan pintas — residual-modulasi v9

| Metadata | Nilai |
| --- | --- |
| ID laporan / versi format | `IMPL-RESMOD-V9-COLAB` / 1.0 |
| Tanggal analisis | 2026-11-26, Asia/Jakarta |
| Rentang kejadian | 2026-11-22 s.d. 2026-11-26 |
| Jenis laporan | Implementasi + catatan insiden |
| Pipeline | `residual_modulation_v9`; branch `colab-local-staging`, digabung ke `master` (fast-forward, commit `3cae177`) |
| Editor / runtime | VS Code dengan kernel Colab; Drive dipasang lewat FUSE; sinkronisasi lokal ↔ Drive dengan rclone |
| Dasar bukti | Output notebook dan diagnosis yang dikirim pengguna, artefak Drive, riwayat git, dan tes `tests/test_pipeline_v9.py` (23 lulus) |

## 1. Ringkasan

Run penuh pertama gagal berkali-kali karena **operasional**, bukan karena metode. Penyebabnya:
- banyak berkas kecil dibaca lewat Drive FUSE;
- manifest terpotong saat ditulis ke Drive;
- `features.csv` ditulis ulang ±1.000 kali, sehingga antrean unggah Drive menumpuk hingga 105 GB di `/root/.config` dan disk penuh;
- kuota Drive akun Colab (akun kampus) penuh.

Pipeline lalu dirancang ulang:
1. Semua kerja berlangsung di disk lokal `/content/work`.
2. Hasil ekstraksi ditulis sebagai *shard* tulis-sekali yang dicadangkan ke Drive.
3. Run baru memulihkan diri dari cadangan.

Setelah perubahan ini, ekstraksi 47.373 utterance, training, stress, dan dua ablation selesai tanpa kehilangan progres. Hasil penelitiannya ada di [laporan hasil 2026-11-26](laporan_hasil_residual_modulation_v9_2026-11-26.md).

Selain infrastruktur, sesi ini menambahkan:
- kontrol jalan pintas (`QC_plus_*`);
- analisis hening (`analyze`);
- penjaga identitas cache fitur;
- desain ablation pita tunggal.

## 2. Kronologi insiden dan perbaikan

| Gejala | Penyebab | Perbaikan |
| --- | --- | --- |
| Audit audio ±5 jam, tidak dapat dilanjutkan | Pemeriksaan berkas satu per satu lewat Drive FUSE | `audit_audio` berthread dengan cache `_cache.csv` dan progres; lalu digantikan *staging* |
| DF 0/812 tersedia | DF hanya ada sebagai `DF.zip` | Ekstraksi langsung dari zip, kemudian `stage` |
| Manifest di Drive terpotong (30.130 dari 47.373 baris) | Penulisan berkas besar ke FUSE tidak selesai sebelum dibaca | Manifest dibangun di disk lokal dengan assert jumlah baris |
| T04 menolak cache (identitas berbeda) | Manifest terpotong mengubah daftar hash | Identitas cache dibuat independen path; manifest lokal |
| Disk penuh; `/root/.config` 105 GB | Antrean unggah Drive dari ±1.000 penulisan ulang `features.csv`, ditambah kuota akun Colab penuh | Rancang ulang: shard tulis-sekali, kerja lokal, preflight kuota (T00b) |
| Unggah penyelamatan `storageQuotaExceeded` | Kuota akun kampus penuh; Drive tetap memakai akun Colab | Runtime direset; folder lama harus dihapus dan Sampah dikosongkan oleh akun kampus |
| `KeyError: 'MFCC_LFCC'` di evaluasi | Variabel loop `b` menimpa variabel champion `b` | Diganti nama menjadi `family_a`/`family_b` |
| Checksum 6 berkas run `1d5ba7b2b9562c28` tidak cocok | Evaluasi ulang menghasilkan float RF berbeda ~1e-16; mirror lama hanya membandingkan ukuran | `mirror_exact` (berbasis hash), perbaikan otomatis saat restore, opsi `RESTORE_RUN_ID` |
| Ablation "tanpa satu pita" tidak mengubah AUC | Energi pita berjumlah ≈ 1, sehingga pita yang dihapus terekonstruksi | `column_families` dan ablation pita tunggal |
| Uji noise SNR 20/10 dB naik tidak monoton | Noise ditambahkan sebelum trim, sehingga trim 35 dB gagal memangkas hening | Didokumentasikan; tidak diperbaiki karena `channels.py` adalah modul identitas cache |

## 3. Rancangan eksekusi yang berlaku

**Folder kerja dan cadangan.**
- Semua berkas kerja berada di `/content/work/<profil>`. T00 memeriksa ruang disk kosong > 30 GB.
- Cadangan di Drive: `PROJECT_ROOT/backup/<profil>/`, berisi `inputs/`, `features/<identitas[:16]>/shards/`, `runs/<id>/`, `checkpoints/<id>/`, `stress/<run>/`, dan `band_ablation/`.

**Staging audio (`src/pipeline_v9/stage.py`).**
- Hanya utterance dalam manifest yang disalin ke `audio/<dataset>/`.
- LA dibaca dari salinan lokal `LA.zip` (dihapus setelah dipakai), DF langsung dari `DF.zip`, dan MLAAD per berkas.
- SHA-256 dihitung dari byte yang ditulis, jadi tidak perlu membaca ulang.
- Path asal disimpan di `source_path`.

**Ekstraksi ber-shard (`src/pipeline_v9/extraction.py`).**
- Setiap 500 utterance menjadi satu shard: CSV baris fitur + tar cache frame `.npz`.
- Ke cadangan, tar ditulis lebih dulu; CSV ditulis terakhir sebagai penanda selesai.
- `restore_backup` memeriksa bahwa isi tar mencakup semua baris `ok` sebelum shard dipakai; shard rusak dilewati.
- Sesi baru memulihkan shard lalu hanya mengekstrak sisanya. Pada runtime baru, T04 harus mencetak `47373 sudah diekstrak; 0 diproses`.

**Identitas cache fitur.** Identitas = hash konten audio + `cfg['features']` + hash lima modul identitas + versi numpy/scipy/librosa/soundfile/soxr/pandas. Nilai aktif: `fb035f69b9a86b98`.
- Lima modul identitas: `audio_io.py`, `channels.py`, `common.py`, `extraction.py`, `features.py`.
- Tes `test_feature_modules_unchanged_or_cache_invalidation_is_deliberate` gagal bila salah satu modul berubah, dengan hash yang dipatok di `tests/feature_code_pins.json`. Ini mencegah ekstraksi ulang ±5 jam yang tidak disengaja.
- Konsekuensinya, keluarga fitur baru (kontrol QC, ablation pita tunggal) didefinisikan di config (`composite_families`, `column_families` di `config.py`), bukan di `features.py`.

**Run dan pemulihan (`runtime.train_and_evaluate`).**
- Setiap seed yang dibekukan langsung dicerminkan ke cadangan dengan `mirror_exact`, yang membandingkan hash, bukan ukuran.
- Run final dan checkpoint juga dicerminkan.
- Bila checksum `complete.json` gagal setelah restore, berkas itu dihapus dan evaluasi dihitung ulang dari seed beku.
- `RESTORE_RUN_ID` di T05 memulihkan run yang sudah selesai tanpa training, untuk sesi yang hanya menjalankan T07/T08.

**Sel notebook.**

| Sel | Fungsi |
| --- | --- |
| T00 | Path dan pemeriksaan ruang disk |
| T00b | Preflight kuota: unggah lalu hapus 1 MB lewat Drive API |
| T01 | Instalasi dependensi |
| T02 | Impor manifest dan profil |
| T03 | Staging audio |
| T04 | Ekstraksi atau pemulihan shard |
| T05 | Training atau pemulihan run |
| T06 | Ringkasan, kontrol QC, dan `analyze` |
| T07 | Stress, dengan cadangan per varian |
| T08 | Ablation pita |

Notebook dibangkitkan oleh `scripts/build_pipeline_v9_notebook.py`. Menjalankan generator **menimpa** output sel notebook lokal. Perubahan teks satu sel sebaiknya ditambal langsung pada JSON notebook.

## 4. Tambahan metode

- **Kontrol jalan pintas.**
  - `QC_plus_Residual_Modulation` dan `QC_plus_MFCC_LFCC` adalah komposit QC + keluarga.
  - `paired_comparisons` menguji keduanya terhadap `QC_only` dengan LR dan HGB (`paired_classifiers`).
- **Analisis hening** (`src/pipeline_v9/analysis.py`, CLI `analyze`), dari paket run tanpa training ulang:
  - AUC tiap kuantitas QC dan durasi hening;
  - ringkasan hening per dataset × split × label;
  - AUC tiap sistem di dalam bin kuantil hening atau durasi.
- **G4** memakai grid modulasi 0,5–20 Hz (80 titik). Interpolasi di luar rentang menjadi NaN; rerata dan bootstrap menangani NaN dan mencatat `n_utterances`.
- **Ablation pita tunggal** (`configs/residual_modulation_v9_band_ablation.yaml`):
  - keluarga: `Modulation_band_*_only`, `Modulation_bands_only`, `Modulation_shape_only`;
  - 12 perbandingan berpasangan;
  - GMM dimatikan.
- **Aset Bab 4** (`scripts/build_bab4_assets.py`): membangkitkan 15 tabel dan 10 gambar dari tiga run setelah memverifikasi checksum. Ketidakcocokan yang diketahui harus disebut eksplisit.

## 5. Sinkronisasi lokal ↔ Drive

- **Remote rclone.**
  - `residual:` (scope drive, `root_folder_id` menunjuk folder proyek).
  - `skripsi:` untuk folder data glottal bersama.
- **Skrip sinkronisasi.** `scripts/rclone_sync.sh upload|download|both [--execute]`: dry-run secara default, `--update`, dan tanpa penghapusan.
- **Filter** (`configs/rclone-sync.filter`): audio, `*.npz`/`*.tar`, `results/_work/**`, `backup/*/features/**`, dan berkas rahasia tidak ditransfer.
- **Pemilik folder.** Folder Drive milik akun pribadi (400 GB). Colab berjalan dengan akun kampus melalui *shortcut*, sehingga revisi berkas dihitung ke kuota akun kampus. Folder lama `results/residual_modulation_v9/full` milik akun kampus harus dihapus oleh akun tersebut.

## 6. Validasi

- `pytest`: 23 tes lulus. Tes yang relevan:
  - staging dari zip/berkas dengan hash byte tertulis;
  - shard, cadangan, restore, dan identitas yang independen path;
  - pin modul identitas;
  - komposit dan `column_families`;
  - `mirror_exact` mengganti berkas berukuran sama dengan isi berbeda;
  - AUC terstratifikasi hening memisahkan jalan pintas dari sinyal.
- Perbaikan otomatis setelah restore diuji manual: cadangan smoke dirusak sengaja, lalu dipulihkan dan dievaluasi ulang. Uji ini tidak termasuk tes otomatis.
- Smoke sintetis untuk config ablation pita tunggal: 12 perbandingan berpasangan terbentuk, dan keluarga satu kolom berjalan di semua classifier serta permutation importance.
- Run nyata:
  - ekstraksi 47.373/47.373 `ok`;
  - run `1d5ba7b2b9562c28` selesai;
  - stress 12/12 varian `ok`;
  - ablation `ac75de17ecbf78eb` dan `52760a671e6853be` dengan 64/64 checksum cocok.

## 7. Batas yang tersisa

- Run `1d5ba7b2b9562c28` masih punya 6 berkas yang checksum-nya tidak cocok dengan `complete.json`. Angkanya konsisten dengan skor tersimpan (§3 laporan hasil). Paket bersih memerlukan T05 tanpa `RESTORE_RUN_ID`.
- `manifests/glottal_v9/*` lokal tertimpa versi Drive dengan path Colab dan tidak di-commit. Untuk memulihkan versi repo:

  ```bash
  git restore manifests/glottal_v9
  ```

- Uji noise setelah trim memerlukan perubahan `channels.py`, sehingga perlu ekstraksi ulang penuh.
