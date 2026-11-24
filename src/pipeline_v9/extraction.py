"""Resumable, shard-based feature extraction with an optional write-once backup (e.g. Drive).

Design (lessons from the Colab runs of 23 Nov 2026):
* Work happens on a fast local folder. Each chunk becomes an immutable shard: one CSV of
  feature rows plus one tar of that chunk's frame caches. Nothing large is ever rewritten,
  so a synced Drive folder does not accumulate revisions or a growing upload queue.
* The backup folder receives each shard once (tar first, CSV last as the commit marker).
  A new session restores the shards, then extracts only what is missing.
* The cache identity depends on audio bytes (SHA-256), feature settings, the modules that
  compute features and the numeric libraries. Paths, manifest layout and training code are
  not part of it, so moving the work folder or fixing reporting code keeps the cache valid.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import secrets
import tarfile
import numpy as np
import pandas as pd
from .audio_io import read_audio
from .baseline_gmm import cache_path
from .common import (copy_file, digest, feature_code_identity, feature_runtime_identity, file_hash,
                     write_csv, write_json)
from .config import load_config
from .data import normalize_manifest
from .features import extract_audio

SHARDS = 'shards'


def _hash_or_missing(path):
    try:
        return file_hash(path)
    except OSError:
        return 'missing'


def verified_hashes(manifest, jobs=16):
    """Hash every input (threads; I/O bound) and refuse silently changed audio."""
    rows = list(manifest.itertuples())
    hashes = {}
    print(f'Verifikasi checksum {len(rows)} audio...', flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for i, (row, actual) in enumerate(zip(rows, pool.map(_hash_or_missing, [r.path for r in rows])), 1):
            if row.sha256 and actual != row.sha256:
                raise ValueError(f'Checksum input berubah/tidak tersedia: {row.utterance_id}')
            hashes[row.utterance_id] = actual
            if i % 5000 == 0 or i == len(rows):
                print(f'  {i}/{len(rows)}', flush=True)
    return hashes


def extraction_identity(cfg, hashes):
    identity = {'features': cfg['features'], 'audio_sha256': hashes,
                'code': feature_code_identity(), 'runtime': feature_runtime_identity()}
    return identity, digest(identity)


def _read_shards(folder):
    files = sorted((Path(folder)/SHARDS).glob('*.csv'))
    if not files:
        return pd.DataFrame(columns=['utterance_id', 'analysis_status'])
    frame = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    # A retried utterance appears again in a later shard; prefer a successful row, then the latest.
    frame['_ok'] = frame.analysis_status.eq('ok')
    frame['_order'] = np.arange(len(frame))
    frame = frame.sort_values(['_ok', '_order']).drop_duplicates('utterance_id', keep='last')
    return frame.sort_values('_order').drop(columns=['_ok', '_order']).reset_index(drop=True)


def restore_backup(out, backup):
    """Bring shards present in the backup but absent locally into `out`. Corrupt shards are skipped."""
    out, backup = Path(out), Path(backup)
    restored = skipped = 0
    (out/SHARDS).mkdir(parents=True, exist_ok=True)
    (out/'frames').mkdir(exist_ok=True)
    for csv in sorted((backup/SHARDS).glob('*.csv')):
        if (out/SHARDS/csv.name).is_file():
            continue
        archive = csv.with_suffix('.tar')
        try:
            rows = pd.read_csv(csv, usecols=['utterance_id', 'analysis_status'])
            names = set()
            if archive.is_file():
                with tarfile.open(archive) as tar:
                    members = [m for m in tar.getmembers() if m.isfile() and '/' not in m.name and m.name.endswith('.npz')]
                    names = {m.name for m in members}
                    tar.extractall(out/'frames', members=members)
            needed = {cache_path(out, u).name for u in rows.loc[rows.analysis_status.eq('ok'), 'utterance_id']}
            if not needed <= names:
                raise ValueError('frame cache tidak lengkap')
            copy_file(csv, out/SHARDS/csv.name)
            restored += 1
        except (OSError, ValueError, tarfile.TarError, pd.errors.ParserError) as exc:
            print(f'  shard cadangan dilewati ({csv.name}): {exc}', flush=True)
            skipped += 1
    print(f'Cadangan: {restored} shard dipulihkan, {skipped} dilewati.', flush=True)


def _backup_shard(out, backup, shard_csv, uids):
    backup_shards = Path(backup)/SHARDS
    backup_shards.mkdir(parents=True, exist_ok=True)
    local_tar = shard_csv.with_suffix('.tar')
    with tarfile.open(local_tar, 'w') as tar:
        for uid in uids:
            path = cache_path(out, uid)
            tar.add(path, arcname=path.name)
    try:
        copy_file(local_tar, backup_shards/local_tar.name)
        copy_file(shard_csv, backup_shards/shard_csv.name)   # commit marker, written last
    except OSError as exc:
        raise OSError(f'Gagal menulis cadangan ke {backup}: {exc}. Periksa kuota Drive akun Colab; '
                      'ekstraksi lokal tetap tersimpan di folder kerja selama runtime hidup.') from exc
    finally:
        local_tar.unlink(missing_ok=True)


def extract_manifest(manifest_path, config_path, out, jobs=1, chunk=500, backup=None):
    from joblib import Parallel, delayed
    cfg = load_config(config_path)
    manifest = normalize_manifest(pd.read_csv(manifest_path, keep_default_na=False))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    hashes = verified_hashes(manifest)
    identity, identity_hash = extraction_identity(cfg, hashes)
    meta_path = out/'extraction_identity.json'
    if meta_path.exists() and json.loads(meta_path.read_text())['identity_hash'] != identity_hash:
        raise ValueError('Identitas cache berbeda. Gunakan folder output baru; cache lama tidak ditimpa.')
    write_json(meta_path, {'identity_hash': identity_hash, **identity})
    if backup is not None:
        # One backup folder per identity: shards of different audio/config/code never mix.
        backup = Path(backup)/identity_hash[:16]
        if (backup/SHARDS).is_dir():
            restore_backup(out, backup)
        backup.mkdir(parents=True, exist_ok=True)
        copy_file(meta_path, backup/'extraction_identity.json')
    (out/'frames').mkdir(exist_ok=True)
    (out/SHARDS).mkdir(exist_ok=True)
    done = _read_shards(out)
    complete = {uid for uid in done.loc[done.analysis_status.eq('ok'), 'utterance_id'] if cache_path(out, uid).is_file()}
    todo = manifest[~manifest.utterance_id.isin(complete)]

    def one(row):
        base = {'utterance_id': row['utterance_id'], 'audio_sha256': hashes[row['utterance_id']]}
        try:
            if base['audio_sha256'] == 'missing':
                raise FileNotFoundError(row['path'])
            raw, sr = read_audio(row['path'])
            if cfg['features'].get('noise_snr_db') is not None or cfg['features'].get('mp3_bitrate') is not None:
                from .channels import transform
                raw, variant_metadata = transform(raw, sr, cfg['features'], row['utterance_id'])
                base.update(variant_metadata)
            features, frames = extract_audio(raw, sr, cfg['features'], return_frames=True)
            target = cache_path(out, row['utterance_id'])
            with target.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream, **frames)
            target.with_suffix('.tmp').replace(target)
            base.update(features, extract_error='')
        except Exception as exc:
            base.update(analysis_status='error', extract_error=f'{type(exc).__name__}: {exc}')
        return base

    print(f'{len(complete)} sudah diekstrak; {len(todo)} diproses (chunk {chunk}, jobs {jobs}).', flush=True)
    for start in range(0, len(todo), chunk):
        rows = Parallel(n_jobs=jobs)(delayed(one)(r) for r in todo.iloc[start:start+chunk].to_dict('records'))
        part = pd.DataFrame(rows)
        shard = out/SHARDS/f'part_{len(list((out/SHARDS).glob("*.csv"))):05d}_{secrets.token_hex(4)}.csv'
        write_csv(shard, part)
        if backup is not None:
            _backup_shard(out, backup, shard, part.loc[part.analysis_status.eq('ok'), 'utterance_id'])
        print(f'{min(start+chunk, len(todo))}/{len(todo)}', flush=True)
    done = _read_shards(out)
    if done.empty:
        raise ValueError('Manifest kosong.')
    write_csv(out/'features.csv', done)
    joined = manifest.merge(done, on='utterance_id', how='left', validate='1:1')
    coverage = joined.groupby(['dataset', 'split', 'label'], dropna=False).agg(
        n=('utterance_id', 'size'), extracted=('analysis_status', lambda s: s.eq('ok').sum())).reset_index()
    coverage['coverage'] = coverage.extracted/coverage.n
    write_csv(out/'coverage.csv', coverage)
    write_csv(out/'failures.csv', joined[~joined.analysis_status.eq('ok')])
    manifest['sha256'] = manifest.utterance_id.map(hashes).replace('missing', '')
    write_csv(out/'manifest.csv', manifest)
    write_json(out/'extraction_complete.json', {
        'feature_table_sha256': file_hash(out/'features.csv'),
        'manifest_sha256': file_hash(out/'manifest.csv'),
        'identity_hash': identity_hash})
    return joined


def load_features(folder, allow_failures=False):
    folder = Path(folder)
    manifest = normalize_manifest(pd.read_csv(folder/'manifest.csv', keep_default_na=False))
    identity = json.loads((folder/'extraction_identity.json').read_text())
    complete = json.loads((folder/'extraction_complete.json').read_text())
    if (complete['feature_table_sha256'] != file_hash(folder/'features.csv') or
        complete['manifest_sha256'] != file_hash(folder/'manifest.csv') or
        complete['identity_hash'] != identity['identity_hash']):
        raise ValueError('Tabel fitur/manifest cache berubah sejak ekstraksi.')
    frame = manifest.merge(pd.read_csv(folder/'features.csv', low_memory=False), on='utterance_id', how='left', validate='1:1')
    invalid = ~frame.analysis_status.eq('ok')
    if invalid.any() and not allow_failures:
        raise ValueError(f'{invalid.sum()} ekstraksi gagal; periksa failures.csv. --allow-failures diperlukan untuk subset eksploratif.')
    if frame.empty:
        raise ValueError('Fitur kosong.')
    return frame[~invalid].reset_index(drop=True), identity, int(invalid.sum())
