"""Stage the selected audio onto fast local disk and hash each file's bytes once.

Reading tens of thousands of small files through a synced Drive mount is slow and fragile.
Staging copies exactly the manifest's utterances into `<out>/audio/<dataset>/`, either from a
zip archive (read sequentially, optionally copied to local disk first) or file by file. The
SHA-256 is computed from the bytes written, so no second pass is needed, and it is what the
extraction cache identity uses. The staged manifest replaces `path` with the local copy and keeps
the original location in `source_path`. Nothing is written to the source location.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import re
import shutil
import zipfile
import pandas as pd
import soundfile as sf
from .common import digest, file_hash, write_csv
from .data import normalize_manifest

SAFE = re.compile(r'^[A-Za-z0-9._-]+$')


def local_name(uid, suffix):
    """Readable when the ID is filename-safe, else a stable digest (IDs may contain slashes)."""
    return f'{uid}{suffix}' if SAFE.match(uid) else f'{digest(uid)[:24]}{suffix}'


def _write(target, payload):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.part')
    temporary.write_bytes(payload)
    temporary.replace(target)
    return hashlib.sha256(payload).hexdigest()


def _probe(row):
    try:
        info = sf.info(row['path'])
        row.update(duration_s=info.duration, sample_rate=info.samplerate, channels=info.channels,
                   audio_status='ready', audio_error='')
    except (OSError, RuntimeError) as exc:
        row.update(audio_status='missing_or_unreadable', audio_error=str(exc))
    return row


def _from_zip(rows, archive_path, out, copy_local):
    archive_path = Path(archive_path)
    pending = [r for r in rows if not Path(r['path']).is_file()]
    if not pending:  # rerun in the same session: files already staged, hash them locally
        for r in rows:
            r['sha256'] = file_hash(r['path'])
        return
    if copy_local:
        local = out/'zips'/archive_path.name
        if not local.is_file() or local.stat().st_size != archive_path.stat().st_size:
            local.parent.mkdir(parents=True, exist_ok=True)
            print(f'  menyalin {archive_path.name} ({archive_path.stat().st_size / 1e9:.1f} GB) ke disk lokal...', flush=True)
            shutil.copyfile(archive_path, local.with_name(local.name + '.part'))
            local.with_name(local.name + '.part').replace(local)
        archive_path = local
    with zipfile.ZipFile(archive_path) as archive:
        index = {}
        for info in archive.infolist():
            if not info.is_dir():
                index.setdefault(info.filename.rsplit('/', 1)[-1], info)
        for i, row in enumerate(rows, 1):
            info = index.get(f"{row['utterance_id']}.flac")
            if info is None:
                row.update(path='', sha256='', audio_status='missing_or_unreadable',
                           audio_error=f'tidak ada di {archive_path.name}')
                continue
            target = Path(row['path'])
            if target.is_file() and target.stat().st_size == info.file_size:
                row['sha256'] = file_hash(target)
            else:
                row['sha256'] = _write(target, archive.read(info))
            if i % 5000 == 0:
                print(f'    {i}/{len(rows)}', flush=True)
    if copy_local:
        archive_path.unlink(missing_ok=True)


def _from_file(row):
    try:
        target, source = Path(row['path']), Path(row['source_path'])
        if target.is_file() and target.stat().st_size == source.stat().st_size:
            row['sha256'] = file_hash(target)
        else:
            row['sha256'] = _write(target, source.read_bytes())
    except OSError as exc:
        row.update(path='', sha256='', audio_status='missing_or_unreadable', audio_error=str(exc))
    return row


def stage_audio(manifest_path, out, zips=None, copy_zip=(), jobs=8, allow_missing=False):
    """zips: {dataset: archive path}; copy_zip: datasets whose archive is copied locally first."""
    zips, copy_zip = dict(zips or {}), set(copy_zip)
    out = Path(out)
    frame = normalize_manifest(pd.read_csv(manifest_path, keep_default_na=False))
    if 'source_path' in frame:
        frame = frame.rename(columns={'source_path': 'glottal_source_path'})
    frame['source_path'] = frame.path
    staged = []
    for dataset, group in frame.groupby('dataset', sort=True):
        rows = group.to_dict('records')
        suffix = '.flac' if dataset in zips else None
        for row in rows:
            name = local_name(row['utterance_id'], suffix or (Path(row['source_path']).suffix.lower() or '.wav'))
            row['path'] = str((out/'audio'/dataset/name).resolve())
            row['audio_status'], row['audio_error'] = 'ready', ''
        print(f'{dataset}: {len(rows)} file dari {zips.get(dataset, "folder sumber")}', flush=True)
        if dataset in zips:
            _from_zip(rows, zips[dataset], out, dataset in copy_zip)
        else:
            with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
                rows = list(pool.map(_from_file, rows))
        staged.extend(rows)
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        staged = list(pool.map(lambda r: _probe(r) if r['audio_status'] == 'ready' else r, staged))
    result = pd.DataFrame(staged).set_index('utterance_id').loc[frame.utterance_id].reset_index()
    write_csv(out/'staged_manifest.csv', result)
    missing = result.audio_status.ne('ready')
    write_csv(out/'staged_missing.csv', result[missing])
    print(f'Audio siap: {int((~missing).sum())}/{len(result)}', flush=True)
    if missing.any() and not allow_missing:
        raise FileNotFoundError(f'{missing.sum()}/{len(result)} audio tidak tersedia; lihat staged_missing.csv. '
                                'Tidak ada penggantian sampel.')
    return result
