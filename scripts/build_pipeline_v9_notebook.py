#!/usr/bin/env python3
"""Build clean local/Colab launchers; source code remains in src/ and scripts/."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'source': text.splitlines(keepends=True), 'execution_count': None, 'outputs': []}


cells = [md(r'''# Pipeline v9 — residual dan modulasi (seluruh utterance)
Acuan alur: **glottal v9 saja**. Panduan residual-modulasi menentukan domain bukti.
Dataset dan split diimpor persis dari manifest run glottal v9. Kode, cache, model, dan hasil milik proyek ini.

**Cara kerja di Colab (dirancang ulang 23 Nov 2026).** Semua proses berat berjalan di disk lokal runtime
(`/content/work`): audio disalin sekali dari zip/folder dataset, fitur dan model ditulis di sana. Drive hanya
menerima *cadangan tulis-sekali* di `backup/`: shard fitur (CSV + tar frame, ±500 utterance per shard), seed
model yang sudah dibekukan, dan paket hasil akhir. Tidak ada file besar yang ditulis ulang di Drive.

Jalankan T00–T06 berurutan. Bila runtime terputus, jalankan ulang dari T00: audio di-*stage* ulang (±15–25 menit),
lalu T04/T05 memulihkan shard dan seed dari `backup/` dan hanya mengerjakan sisanya.
'''), code(r'''# T00 — lokasi proyek dan sumber dataset. Ubah hanya blok ini.
from pathlib import Path
import os, sys, subprocess, shutil, json
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_ROOT = Path('/content/drive/MyDrive/College/speech-deepfake-residual-modulation-new')
    GLOTTAL_ROOT = Path('/content/drive/MyDrive/College/speech-deepfake-glottal')
    WORK_ROOT = Path('/content/work')            # disk lokal runtime: semua proses berat di sini
else:
    PROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'configs/residual_modulation_v9.yaml').exists())
    GLOTTAL_ROOT = PROJECT_ROOT.parent/'speech-deepfake-glottal'
    WORK_ROOT = PROJECT_ROOT/'results/_work'
DATA_ROOT = GLOTTAL_ROOT/'data'
SOURCE_RUN = GLOTTAL_ROOT/'results/forensic_whole_utterance_v9/fe17e2b856dd5d6c'
PROFILE = 'full'       # full = pilihan v9 persis; smoke = 1 utterance/sel, subset eksplisit
N_JOBS = os.cpu_count() or 2                     # tidak memengaruhi identitas cache
BACKUP = PROJECT_ROOT/'backup'/PROFILE            # satu-satunya tujuan tulis di Drive
CODE_ROOT = PROJECT_ROOT
RUNNER = CODE_ROOT/'scripts/run_pipeline_v9.py'
assert RUNNER.is_file(), f'Salin src/, scripts/, configs/, requirements.txt ke {CODE_ROOT}'
assert PROJECT_ROOT.resolve() != GLOTTAL_ROOT.resolve(), 'Output wajib berada pada proyek residual-modulasi.'
WORK_ROOT.mkdir(parents=True, exist_ok=True)
free_gb = shutil.disk_usage(WORK_ROOT).free / 1e9
print(f'Kode: {PROJECT_ROOT}\nData: {DATA_ROOT}\nKerja lokal: {WORK_ROOT} ({free_gb:.0f} GB kosong)\nCadangan: {BACKUP}\nN_JOBS={N_JOBS}')
assert free_gb > 30, 'Disk lokal < 30 GB; gunakan runtime baru.'
'''), code(r'''# T00b — cek Drive akun Colab SEBELUM bekerja berjam-jam (Colab saja).
# File yang ditulis Colab memakai kuota akun yang login di Colab, walau foldernya milik akun lain.
# Uji ini mengunggah 1 MB lalu menghapusnya; kuota penuh langsung terdeteksi (storageQuotaExceeded).
if IN_COLAB:
    from google.colab import auth
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaInMemoryUpload
    auth.authenticate_user()
    api = build('drive', 'v3')
    me = api.about().get(fields='user(emailAddress)').execute()['user']['emailAddress']
    try:
        test = api.files().create(body={'name': '.colab_quota_test'}, media_body=MediaInMemoryUpload(os.urandom(1 << 20)),
                                  fields='id').execute()
        api.files().delete(fileId=test['id']).execute()
    except HttpError as error:
        raise SystemExit(f'Drive akun Colab ({me}) menolak file baru: {error}.\n'
                         'Kosongkan kuota akun ini (termasuk Sampah) atau login Colab dengan akun pemilik folder.')
    shortcut = '.shortcut-targets-by-id' in str(PROJECT_ROOT.resolve())
    print(f'Akun Colab: {me} — unggah uji OK.' + (' Folder proyek adalah shortcut milik akun lain; cadangan tetap '
          'memakai kuota akun Colab (±8–10 GB untuk run penuh).' if shortcut else ''))
'''), code(r'''# T01 — environment
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(CODE_ROOT/'requirements.txt')], check=True)
os.chdir(CODE_ROOT)
sys.path.insert(0, str(CODE_ROOT))
def run(*args):
    subprocess.run([sys.executable, str(RUNNER), *map(str, args)], check=True, cwd=CODE_ROOT)
run('--help')
'''), md(r'''## T02 — manifest yang sama dengan glottal v9
Importer hanya membaca `manifests/*.csv` dari run v9 terpilih (ID, label, dataset, split dipertahankan).
Manifest dibuat di disk lokal dan jumlah barisnya diperiksa; salinannya disimpan ke `backup/inputs`.
'''), code(r'''import pandas as pd
import yaml
from src.pipeline_v9.common import write_csv, write_json, file_hash, mirror_tree
MANIFESTS = WORK_ROOT/'manifests'
run('import-glottal-v9', '--source-run', SOURCE_RUN, '--data-root', DATA_ROOT, '--out', MANIFESTS)
manifest = pd.read_csv(MANIFESTS/'manifest.csv', keep_default_na=False)
expected = sum(v['rows'] for v in json.loads((MANIFESTS/'source_v9.json').read_text())['manifests'].values())
assert len(manifest) == expected, f'Manifest {len(manifest)} baris, seharusnya {expected}.'
def apply_profile(cfg):
    if PROFILE == 'smoke':
        cfg.update(seeds=[2026], bootstrap_draws=100, permutation_repeats=3)
        cfg['gmm'].update(n_components=2, max_frames=20000)
    elif PROFILE != 'full':
        raise ValueError('PROFILE harus full atau smoke')
    return cfg
settings = apply_profile(yaml.safe_load((CODE_ROOT/'configs/residual_modulation_v9.yaml').read_text()))
if PROFILE == 'smoke':
    manifest = pd.concat([g.sample(1, random_state=2026) for _, g in manifest.groupby(
        ['dataset', 'split', 'speaker_id', 'label', 'attack_id'])], ignore_index=True)
WORK = WORK_ROOT/PROFILE
INPUTS = WORK/'inputs'
MANIFEST, CONFIG = INPUTS/'input_manifest.csv', INPUTS/'config.yaml'
write_csv(MANIFEST, manifest)
CONFIG.write_text(yaml.safe_dump(settings, sort_keys=False))
shutil.copyfile(MANIFESTS/'source_v9.json', INPUTS/'source_v9.json')
write_json(INPUTS/'selection.json', {'profile': PROFILE, 'source_run': str(SOURCE_RUN),
    'source_manifest_sha256': file_hash(MANIFESTS/'manifest.csv'), 'n': len(manifest)})
mirror_tree(INPUTS, BACKUP/'inputs')
print(manifest.groupby(['dataset', 'split', 'label']).size())
print('Total:', len(manifest))
'''), md(r'''## T03 — stage audio ke disk lokal + audit
LA 2019 dibaca dari `LA.zip` yang disalin ke disk lokal (satu bacaan berurutan 7,6 GB jauh lebih cepat daripada
44 ribu file kecil di Drive). DF 2021 dibaca langsung dari `DF.zip` (hanya 812 file). MLAAD disalin per file.
SHA-256 dihitung dari byte yang ditulis, lalu dicek kebocoran speaker/audio. Tidak ada yang ditulis ke folder glottal.
'''), code(r'''STAGE = WORK_ROOT/'stage'
run('stage', '--manifest', MANIFEST, '--out', STAGE,
    '--zip', f"asvspoof2019_la={DATA_ROOT/'external/asvspoof2019_la/LA.zip'}",
    '--zip', f"asvspoof2021_df={DATA_ROOT/'external/asvspoof2021_df/DF.zip'}",
    *(['--copy-zip', 'asvspoof2019_la'] if PROFILE == 'full' else []), '--jobs', 16)
STAGED = STAGE/'staged_manifest.csv'
staged = pd.read_csv(STAGED, keep_default_na=False)
assert len(staged) == len(manifest) and staged.audio_status.eq('ready').all()
for name in ['staged_manifest.csv', 'staged_manifest.json']:
    shutil.copyfile(STAGE/name, INPUTS/name)
mirror_tree(INPUTS, BACKUP/'inputs')
print('Audio siap:', len(staged), '| disk kosong:', f'{shutil.disk_usage(WORK_ROOT).free / 1e9:.0f} GB')
'''), code(r'''# T04 — ekstraksi seluruh utterance. Shard tulis-sekali dicadangkan ke Drive dan dipulihkan saat dilanjutkan.
FEATURES = WORK/'features'
run('extract', '--manifest', STAGED, '--config', CONFIG, '--out', FEATURES,
    '--jobs', N_JOBS, '--chunk', 500, '--backup', BACKUP/'features')
display(pd.read_csv(FEATURES/'coverage.csv'))
'''), code(r'''# T05 — train/dev, bekukan semua seed, baru evaluasi target dan laporan G1–G7.
# Seed yang sudah beku langsung dicadangkan; paket hasil disalin ke Drive setelah selesai.
RUNS, CHECKPOINTS = WORK/'runs', WORK/'checkpoints'
# Sesi baru hanya untuk T07/T08? Isi RUN_ID run yang sudah selesai di Drive (mis. '1d5ba7b2b9562c28'):
# paket hasil dan seed beku disalin dari backup tanpa training/evaluasi ulang. T04 tetap perlu dijalankan.
RESTORE_RUN_ID = None
if RESTORE_RUN_ID:
    mirror_tree(BACKUP/'runs'/RESTORE_RUN_ID, RUNS/RESTORE_RUN_ID)
    mirror_tree(BACKUP/'checkpoints'/RESTORE_RUN_ID, CHECKPOINTS/RESTORE_RUN_ID)
    RUN_ID = RESTORE_RUN_ID
else:
    run('run', '--features', FEATURES, '--config', CONFIG, '--out', RUNS, '--checkpoints', CHECKPOINTS, '--backup', BACKUP)
    RUN_ID = (RUNS/'LATEST_RUN').read_text().strip()
RUN_DIR = RUNS/RUN_ID
print('Run:', RUN_ID)
'''), code(r'''# T06 — ringkasan; SD antarseed berbeda dari CI bootstrap speaker.
assert (RUN_DIR/'complete.json').is_file(), f'Run belum lengkap: {RUN_DIR}'
display(pd.read_csv(RUN_DIR/'seed_summary.csv'))
display(pd.read_csv(RUN_DIR/'evidence_consistency.csv'))
# Kontrol shortcut: selisih AUC berpasangan (bootstrap speaker, seed pertama). CI tidak memuat 0 = berbeda.
ci = pd.read_csv(RUN_DIR/'bootstrap_intervals.csv')
if 'comparison' in ci:
    controls = ci[ci.comparison.notna() & ci.status.eq('paired_difference') & ci.metric.eq('auc')
                  & ci.seed.eq(settings['seeds'][0])]
    display(controls[['scenario', 'comparison', 'median', 'low', 'high']].round(3))
# Analisis shortcut hening/durasi (tanpa training ulang): tabel di RUN_DIR/analysis, disalin ke Drive.
run('analyze', '--run', RUN_DIR)
mirror_tree(RUN_DIR/'analysis', BACKUP/'runs'/RUN_ID/'analysis')
strat = pd.read_csv(RUN_DIR/'analysis'/'stratified_auc_summary.csv')
display(strat[strat.split.ne('dev')].pivot_table(index='system', columns=['dataset', 'split'],
        values=['pooled_auc_mean', 'auc_within_silence_seconds_mean']).round(3))
print('Hasil lokal:', RUN_DIR)
print('Hasil di Drive:', BACKUP/'runs'/RUN_ID)
'''), md(r'''## T07 — sensitivitas setelah model dibekukan (opsional)
Mengubah satu faktor per varian (orde LPC 12/20, trim 25/45 dB/tanpa trim, noise SNR 30/20/10, MP3 128/64/32 kbps)
pada subset target yang tetap: `per-cell` utterance per dataset × split × speaker × label × serangan.
`per-cell 1` = 1.294 utterance × 12 varian (±1,5 jam pada 2 core); `per-cell 2` = 2.336 utterance (±2,5–3 jam).
Model seed pertama dan threshold tetap; GMM tidak disertakan. Ini menguji sensitivitas pengukuran, bukan tuning target.
Shard fitur tiap varian dan tabel hasil dicadangkan ke `backup/<PROFILE>/stress/<RUN_ID>/`; menjalankan ulang melanjutkan.
'''), code(r'''RUN_STRESS = False     # ubah ke True untuk menjalankan
STRESS_PER_CELL = 1    # 1 ≈ separuh waktu dari 2
if RUN_STRESS:
    run('stress', '--run', RUN_DIR, '--features', FEATURES, '--checkpoints', CHECKPOINTS,
        '--per-cell', STRESS_PER_CELL, '--jobs', N_JOBS, '--backup', BACKUP/'stress'/RUN_ID)
    display(pd.read_csv(RUN_DIR/'stress'/'variant_status.csv'))
    metrics = pd.read_csv(RUN_DIR/'stress'/'metrics.csv')
    display(metrics.pivot_table(index='variant', columns=['dataset', 'system'], values='auc').round(3))
'''), md(r'''## T08 — ablation pita modulasi (opsional, ±1–2 jam training, tanpa ekstraksi ulang)
Melatih ulang dengan `configs/residual_modulation_v9_band_ablation.yaml`: tiap pita modulasi **sendirian**, keempat pita
tanpa centroid/entropy/variabilitas, dan fitur bentuk tanpa pita. Energi pita berjumlah ≈1, jadi "tanpa satu pita"
tidak informatif (bisa direkonstruksi dari tiga sisanya). Fitur memakai cache T04 yang sama; GMM dimatikan.
Eksperimen development terpisah; jangan memilih konfigurasi dari hasil eksternal.
'''), code(r'''RUN_ABLATION = False   # ubah ke True untuk menjalankan
if RUN_ABLATION:
    ABLATION_CONFIG = INPUTS/'config_band_ablation.yaml'
    ABLATION_CONFIG.write_text(yaml.safe_dump(apply_profile(yaml.safe_load(
        (CODE_ROOT/'configs/residual_modulation_v9_band_ablation.yaml').read_text())), sort_keys=False))
    mirror_tree(INPUTS, BACKUP/'inputs')
    ABLATION_RUNS = WORK/'runs_band_ablation'
    run('run', '--features', FEATURES, '--config', ABLATION_CONFIG, '--out', ABLATION_RUNS,
        '--checkpoints', WORK/'checkpoints_band_ablation', '--backup', BACKUP/'band_ablation')
    ABLATION_DIR = ABLATION_RUNS/(ABLATION_RUNS/'LATEST_RUN').read_text().strip()
    summary = pd.read_csv(ABLATION_DIR/'seed_summary.csv')
    display(summary[summary.system.str.startswith('Modulation')].pivot_table(
        index='system', columns='scenario', values='auc_mean').round(3))
    print('Hasil ablation di Drive:', BACKUP/'band_ablation'/'runs'/ABLATION_DIR.name)
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
Jalankan notebook utama T00–T03 untuk impor manifest, staging audio, dan audit. File yang hilang harus disediakan pada path audit,
bukan diganti dengan utterance lain. DEEP-VOICE opsional membutuhkan metadata speaker terverifikasi sebelum group split.
'''), code('''from pathlib import Path
import subprocess, sys
PROJECT_ROOT = Path.cwd()  # ubah ke folder proyek ini bila diperlukan
GLOTTAL_ROOT = PROJECT_ROOT.parent/'speech-deepfake-glottal'
subprocess.run([sys.executable, str(PROJECT_ROOT/'scripts/run_pipeline_v9.py'), 'import-glottal-v9',
    '--source-run', str(GLOTTAL_ROOT/'results/forensic_whole_utterance_v9/fe17e2b856dd5d6c'),
    '--data-root', str(GLOTTAL_ROOT/'data'), '--out', str(PROJECT_ROOT/'manifests/glottal_v9')], check=True)
''')])
