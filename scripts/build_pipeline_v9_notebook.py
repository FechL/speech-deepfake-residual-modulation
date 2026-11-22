#!/usr/bin/env python3
"""Build clean local/Colab launchers; source code remains in src/ and scripts/."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'source': text.splitlines(keepends=True), 'execution_count': None, 'outputs': []}


cells = [md('''# Pipeline v9 — residual dan modulasi (seluruh utterance)
Acuan alur: **glottal v9 saja**. Panduan residual-modulasi menentukan domain bukti.
Dataset dan split dapat diimpor persis dari manifest run glottal v9. Kode, cache, model, dan hasil milik proyek ini.

Jalankan T00–T06 berurutan. Setelah runtime terputus, jalankan ulang dengan path yang sama: ekstraksi dapat dilanjutkan.
Audio yang tidak tersedia dicatat, tidak diganti otomatis. Kode proyek harus sudah disalin ke `PROJECT_ROOT` di Drive.
'''), code('''# T00 — lokasi proyek dan sumber dataset. Ubah hanya blok ini.
from pathlib import Path
import os, sys, subprocess
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_ROOT = Path('/content/drive/MyDrive/College/speech-deepfake-residual-modulation-new')
    GLOTTAL_ROOT = Path('/content/drive/MyDrive/College/speech-deepfake-glottal')
else:
    PROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'configs/residual_modulation_v9.yaml').exists())
    GLOTTAL_ROOT = PROJECT_ROOT.parent/'speech-deepfake-glottal'
CODE_ROOT = PROJECT_ROOT
DATA_ROOT = GLOTTAL_ROOT/'data'
SOURCE_RUN = GLOTTAL_ROOT/'results/forensic_whole_utterance_v9/fe17e2b856dd5d6c'
PROFILE = 'full'       # full = exact imported v9 selection; smoke = 1 utterance/cell, explicitly a subset
N_JOBS = 2
RUNNER = CODE_ROOT/'scripts/run_pipeline_v9.py'
assert RUNNER.is_file(), f'Salin src/, scripts/, configs/, requirements.txt ke {CODE_ROOT}'
assert PROJECT_ROOT.resolve() != GLOTTAL_ROOT.resolve(), 'Output wajib berada pada proyek residual-modulasi.'
print('Kode/hasil:', PROJECT_ROOT, '\\nData:', DATA_ROOT)
'''), code('''# T01 — environment
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(CODE_ROOT/'requirements.txt')], check=True)
os.chdir(CODE_ROOT)
sys.path.insert(0, str(CODE_ROOT))
def run(*args):
    subprocess.run([sys.executable, str(RUNNER), *map(str, args)], check=True, cwd=CODE_ROOT)
run('--help')
'''), md('''## T02 — manifest yang sama dengan glottal v9
Importer hanya membaca `manifests/*.csv` dari run v9 terpilih dan mengabaikan `_la_protocol_full.csv`.
ID, label, dataset, dan split dipertahankan; path runtime lama dipetakan ke `DATA_ROOT`.
Untuk dataset baru, buat manifest lewat CLI `manifest-la` atau `manifest-folder`, lalu gunakan schema yang sama.
'''), code('''import pandas as pd
import yaml
from src.pipeline_v9.common import write_csv, write_json, file_hash
MANIFEST_ROOT = PROJECT_ROOT/'manifests/glottal_v9'
run('import-glottal-v9', '--source-run', SOURCE_RUN, '--data-root', DATA_ROOT, '--out', MANIFEST_ROOT)
manifest = pd.read_csv(MANIFEST_ROOT/'manifest.csv', keep_default_na=False)
settings = yaml.safe_load((CODE_ROOT/'configs/residual_modulation_v9.yaml').read_text())
WORK = PROJECT_ROOT/'results/residual_modulation_v9'/PROFILE
WORK.mkdir(parents=True, exist_ok=True)
if PROFILE == 'smoke':
    manifest = pd.concat([g.sample(1, random_state=2026) for _, g in manifest.groupby(
        ['dataset', 'split', 'speaker_id', 'label', 'attack_id'])], ignore_index=True)
    settings.update(seeds=[2026], bootstrap_draws=100, permutation_repeats=3)
    settings['gmm'].update(n_components=2, max_frames=20000)
elif PROFILE != 'full':
    raise ValueError('PROFILE harus full atau smoke')
MANIFEST = WORK/'input_manifest.csv'
CONFIG = WORK/'config.yaml'
write_csv(MANIFEST, manifest)
CONFIG.write_text(yaml.safe_dump(settings, sort_keys=False))
write_json(WORK/'selection.json', {'profile': PROFILE, 'source_run': str(SOURCE_RUN),
    'source_manifest_sha256': file_hash(MANIFEST_ROOT/'manifest.csv'), 'n': len(manifest)})
print(manifest.groupby(['dataset', 'split', 'label']).size())
'''), code('''# T03 — audit audio dan leakage. Berhenti bila input belum lengkap.
AUDITED = WORK/'audited_manifest.csv'
run('audit', '--manifest', MANIFEST, '--out', AUDITED)
'''), code('''# T04 — ekstraksi resumable seluruh utterance; cache dipisah menurut identitas config/audio/kode.
FEATURE_DIR = WORK/'features'
run('extract', '--manifest', AUDITED, '--config', CONFIG, '--out', FEATURE_DIR, '--jobs', N_JOBS)
display(pd.read_csv(FEATURE_DIR/'coverage.csv'))
'''), code('''# T05 — train/dev, freeze semua seed, baru evaluasi target dan laporan G1–G7.
CHECKPOINTS = PROJECT_ROOT/'checkpoints/residual_modulation_v9'
RUNS = WORK/'runs'
run('run', '--features', FEATURE_DIR, '--config', CONFIG, '--out', RUNS, '--checkpoints', CHECKPOINTS)
'''), code('''# T06 — ringkasan; SD antarseed berbeda dari CI bootstrap speaker.
import json
from src.pipeline_v9.common import digest, code_identity, runtime_identity
feature_identity = json.loads((FEATURE_DIR/'extraction_identity.json').read_text())
identity = {'config': settings, 'feature_identity': feature_identity['identity_hash'],
    'code': code_identity(), 'runtime': runtime_identity(),
    'feature_table_sha256': file_hash(FEATURE_DIR/'features.csv'),
    'manifest_sha256': file_hash(FEATURE_DIR/'manifest.csv'), 'allow_failures': False}
RUN_DIR = RUNS/digest(identity)[:16]
assert (RUN_DIR/'complete.json').is_file(), f'Run belum lengkap: {RUN_DIR}'
display(pd.read_csv(RUN_DIR/'seed_summary.csv'))
display(pd.read_csv(RUN_DIR/'evidence_consistency.csv'))
print('Hasil:', RUN_DIR)
print('Gambar:', RUN_DIR/'figures')
'''), md('''## T07 — sensitivitas opsional setelah model dibekukan
Mengubah satu faktor (orde LPC 12/20, trimming, noise) pada subset target yang tetap.
Semua model dan threshold tetap; GMM tidak disertakan. Ini menguji sensitivitas pengukuran, bukan tuning target.
Untuk ablation fitur pita modulasi yang dilatih ulang, gunakan config `residual_modulation_v9_band_ablation.yaml`
sebagai eksperimen development terpisah sebelum evaluasi eksternal final.
'''), code('''# Jalankan jika diperlukan:
# run('stress', '--run', RUN_DIR, '--features', FEATURE_DIR,
#     '--checkpoints', CHECKPOINTS, '--per-cell', 2, '--jobs', N_JOBS)
''')]


def write_notebook(path, cells):
    for i, cell in enumerate(cells):
        cell['id'] = f'v9-{i:03d}'
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    write_notebook(ROOT/'notebooks/pipeline_forensic_whole_utterance_v9.ipynb', cells)
    write_notebook(ROOT/'notebooks/setup/setup_dataset_v9.ipynb', [md('''# Setup dataset v9
Gunakan dataset yang sama dengan glottal v9. Tidak ada pengunduhan atau perubahan audio sumber.
Jalankan notebook utama T00–T03 untuk impor manifest dan audit. File yang hilang harus disediakan pada path audit,
bukan diganti dengan utterance lain. DEEP-VOICE opsional membutuhkan metadata speaker terverifikasi sebelum group split.
'''), code('''from pathlib import Path
import subprocess, sys
PROJECT_ROOT = Path.cwd()  # ubah ke folder proyek ini bila diperlukan
GLOTTAL_ROOT = PROJECT_ROOT.parent/'speech-deepfake-glottal'
subprocess.run([sys.executable, str(PROJECT_ROOT/'scripts/run_pipeline_v9.py'), 'import-glottal-v9',
    '--source-run', str(GLOTTAL_ROOT/'results/forensic_whole_utterance_v9/fe17e2b856dd5d6c'),
    '--data-root', str(GLOTTAL_ROOT/'data'), '--out', str(PROJECT_ROOT/'manifests/glottal_v9')], check=True)
''')])
