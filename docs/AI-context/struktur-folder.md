# Struktur proyek residual-modulasi v9

Struktur tingkat atas mengikuti proyek glottal, dengan hanya pipeline v9 aktif:

- `data/raw`, `data/processed`, `data/external`, `data/primary`, `data/sample`, `data/demo`: tempat input/metadata data; sumber audio boleh berada di proyek dataset glottal.
- `manifests/glottal_v9`: salinan seleksi ID/split, provenance hash, dan audit path. Manifest ini milik proyek residual-modulasi.
- `src/pipeline_v9`: seluruh implementasi independen.
- `scripts/run_pipeline_v9.py`: CLI; `build_pipeline_v9_notebook.py`: generator launcher.
- `configs`: pipeline utama dan ablation fitur pita.
- `notebooks/pipeline_forensic_whole_utterance_v9.ipynb`: launcher utama; `notebooks/setup`: persiapan dataset.
- `results/residual_modulation_v9`: cache, hasil utama per run, dan stress.
- `checkpoints/residual_modulation_v9`: model beku per run/seed.
- `figures`, `exports`: tujuan ekspor; gambar sumber tetap di dalam hasil per run.
- `docs/AI-context`, `docs/AI-research`, `docs/pustaka`, `docs/proposal`: orientasi, laporan, pustaka dan proposal.
- `tests`: verifikasi fitur, protokol, cache, dan kebocoran.

Tidak ada penyalinan archive atau implementasi v1–v8. `.git`, `.agents`, `.aws`, dan `.codex` lingkungan tidak diubah. Folder glottal tidak dimodifikasi.
