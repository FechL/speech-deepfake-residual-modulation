# Indeks laporan penelitian — residual-modulasi

Laporan mengikuti struktur format 1.0 proyek glottal ([template](../../../speech-deepfake-glottal/docs/AI-research/template_laporan_hasil.md)). Cantumkan dataset aktual, RUN_ID, sumber bukti, status eksekusi, dan batas kesimpulan yang terverifikasi. Rumusan masalah ada di [panduan skripsi](../panduan.pdf) §2; keputusan protokol ada di [pemetaan v9](../AI-context/pemetaan-v9-panduan.md).

## Laporan hasil

| Tanggal analisis | Laporan | Skenario / identitas | Format |
| --- | --- | --- | --- |
| 2026-11-26 | [Hasil ASVspoof 2019 LA, 2021 DF, dan MLAAD-tiny — residual-modulasi v9](laporan_hasil_residual_modulation_v9_2026-11-26.md) | Run `1d5ba7b2b9562c28` (profil full, tiga seed; 47.373 utt, cakupan 100%), stress 12 kondisi, ablation pita tunggal `52760a671e6853be`. RM AUC 0,729/0,754 di LA, 0,673/0,663 di DF, di bawah MFCC_LFCC (0,862 LA). Durasi hening sendiri AUC 0,89 (LA); RM tetap 0,65–0,71 di dalam desil hening. RM jauh lebih tahan MP3 (−0,05 vs −0,20), peka trim; modulasi bekerja lewat entropy/bentuk, bukan energi pita; konsistensi dengan baseline κ ≤ 0,07 pada serangan baru | 1.0 |

## Implementasi dan catatan teknis

| Tanggal dokumen | Dokumen | Jenis |
| --- | --- | --- |
| 2026-11-26 | [Implementasi eksekusi Colab, cadangan, dan kontrol jalan pintas](laporan_implementasi_colab_staging_v9_2026-11-26.md) | Kronologi insiden Colab/Drive, rancang ulang shard tulis-sekali + staging lokal, identitas cache fitur, pemulihan run, kontrol QC, analisis hening, ablation pita tunggal |
| 2026-11-22 | [Implementasi pipeline residual dan modulasi v9](laporan_implementasi_residual_modulation_v9_2026-11-22.md) | Pipeline awal, impor manifest glottal v9, validasi sintetis; belum ada audio nyata (historis) |

Laporan lama mempertahankan keadaan bukti pada tanggalnya. Pernyataan "audio belum tersedia" pada laporan 22 November sudah digantikan hasil run penuh 26 November.

## Aset

Artefak pendamping berada di `assets/<pipeline>_<RUN_ID>_<tanggal>/`. Folder [residual_modulation_v9_1d5ba7b2b9562c28_2026-11-26](assets/residual_modulation_v9_1d5ba7b2b9562c28_2026-11-26/) memuat:
- `tabel_bab4.md`, berisi 15 tabel Bab 4;
- `tables/*.csv`;
- `figures/*.png`, berisi 10 gambar;
- `sha256_sumber.txt`.

Semuanya dibangkitkan oleh `scripts/build_bab4_assets.py` dari run di `backup/full/` (salinan rclone cadangan Drive). Jangan mengedit angka secara manual; bangkitkan ulang. Audio, cache fitur, dan model biner tidak termasuk aset.
