"""Protocol-first manifests; no inferred speakers from arbitrary folder names."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import GroupShuffleSplit
from .common import file_hash, write_csv, write_json

AUDIO_EXT = {'.wav', '.flac', '.mp3', '.ogg', '.m4a', '.opus'}
REQUIRED = ['utterance_id', 'path', 'label', 'speaker_id', 'attack_id', 'split', 'dataset']


def normalize_manifest(frame):
    frame = frame.copy()
    for old, new in [('file_path', 'path'), ('generator_id', 'attack_id'), ('audio_sha256', 'sha256')]:
        if new not in frame and old in frame:
            frame[new] = frame[old]
    missing = set(REQUIRED) - set(frame)
    if missing:
        raise ValueError(f'Kolom manifest kurang: {sorted(missing)}')
    for column in REQUIRED:
        if frame[column].isna().any():
            raise ValueError(f'Kolom {column} berisi nilai kosong.')
    frame['label'] = pd.to_numeric(frame.label, errors='raise')
    if not frame.label.isin([0, 1]).all():
        raise ValueError('Label wajib real=0 dan deepfake=1.')
    frame['label'] = frame.label.astype(int)
    frame['split'] = frame.split.replace({'validation': 'dev', 'test': 'eval'})
    if not frame.split.isin(['train', 'dev', 'eval', 'external', 'unassigned']).all():
        raise ValueError('Split tidak dikenal.')
    if frame.utterance_id.duplicated().any():
        raise ValueError('utterance_id harus unik, termasuk lintas dataset.')
    for c in ['sha256', 'source_id']:
        if c not in frame:
            frame[c] = ''
        frame[c] = frame[c].fillna('')
    return frame


def leakage_audit(frame, require_development=True):
    frame = normalize_manifest(frame)
    dev = frame[frame.split.isin(['train', 'dev'])]
    if require_development:
        for split in ['train', 'dev']:
            part = dev[dev.split.eq(split)]
            if part.empty or part.label.nunique() != 2:
                raise ValueError(f'{split} wajib memuat dua kelas.')
        if dev.speaker_id.astype(str).str.lower().str.contains(r'unknown|unassigned|fill_from').any():
            raise ValueError('Speaker development belum diketahui; tidak bisa mengklaim split independen.')
    for key in ['path', 'sha256', 'source_id']:
        known = frame[frame[key].ne('')]
        # No test audio or original-recording source may enter development.
        if key == 'source_id':
            keys = [frame.loc[known.index, 'dataset'], known[key]]
            overlap = known.groupby(keys).split.nunique()
        else:
            overlap = known.groupby(key).split.nunique()
        if (overlap > 1).any():
            raise ValueError(f'Leakage lintas split pada {key}: {list(overlap[overlap > 1].index)[:5]}')
    internal = frame[~frame.split.eq('external')]
    if (internal.groupby(['dataset', 'speaker_id']).split.nunique() > 1).any():
        raise ValueError('Speaker muncul pada lebih dari satu split internal.')
    shared = set(dev.speaker_id) & set(frame.loc[frame.split.eq('external'), 'speaker_id'])
    return {'status': 'ok', 'rows': len(frame), 'internal_speaker_disjoint': True,
            'external_shared_development_speakers': sorted(shared),
            'external_note': 'Cross-dataset does not imply unseen speakers; overlap reported explicitly.'}


def audit_audio(frame, out, allow_missing=False):
    rows = []
    for row in normalize_manifest(frame).to_dict('records'):
        try:
            path = Path(row['path'])
            row['sha256'] = file_hash(path)
            info = sf.info(path)
            row.update(duration_s=info.duration, sample_rate=info.samplerate, channels=info.channels,
                       audio_status='ready', audio_error='')
        except (OSError, RuntimeError) as exc:
            row.update(audio_status='missing_or_unreadable', audio_error=str(exc))
        rows.append(row)
    result = pd.DataFrame(rows)
    write_csv(out, result)
    missing = result.audio_status.ne('ready')
    write_csv(Path(out).with_name(Path(out).stem + '_missing.csv'), result[missing])
    if missing.any() and not allow_missing:
        raise FileNotFoundError(f'{missing.sum()}/{len(result)} audio tidak tersedia; lihat audit missing. Tidak ada penggantian sampel.')
    return result


def manifest_la(root):
    root = Path(root)
    rows = []
    for split, suffix in [('train', 'train.trn'), ('dev', 'dev.trl'), ('eval', 'eval.trl')]:
        protocol = root/'ASVspoof2019_LA_cm_protocols'/f'ASVspoof2019.LA.cm.{suffix}.txt'
        for line in protocol.read_text().splitlines():
            if not line.strip():
                continue
            speaker, uid, _, attack, key = line.split()
            if key not in ['bonafide', 'spoof']:
                raise ValueError(f'Label protokol tidak dikenal: {key}')
            rows.append(dict(utterance_id=uid, label=int(key == 'spoof'), speaker_id=speaker,
                attack_id=attack if key == 'spoof' else 'bonafide', split=split, dataset='asvspoof2019_la',
                path=str(root/f'ASVspoof2019_LA_{split}'/'flac'/f'{uid}.flac')))
    return normalize_manifest(pd.DataFrame(rows))


def manifest_folder(root, dataset, split='external', metadata=None):
    root = Path(root)
    rows = []
    for label, folder in [(0, 'REAL'), (1, 'FAKE')]:
        for path in sorted((root/folder).rglob('*')):
            if path.is_file() and path.suffix.lower() in AUDIO_EXT:
                relative = path.relative_to(root).as_posix()
                rows.append(dict(utterance_id=f'{dataset}/{relative}', path=str(path.resolve()),
                    relative_path=relative, label=label, dataset=dataset, split=split,
                    speaker_id='unknown', attack_id='bonafide' if label == 0 else 'unknown'))
    if not rows:
        raise FileNotFoundError(f'Tidak ada audio REAL/FAKE di {root}.')
    frame = pd.DataFrame(rows)
    if metadata:
        meta = pd.read_csv(metadata, keep_default_na=False)
        if 'relative_path' not in meta or meta.relative_path.duplicated().any():
            raise ValueError('Metadata harus mempunyai relative_path unik.')
        for column in ['speaker_id', 'attack_id', 'source_id']:
            if column in meta:
                mapping = meta.set_index('relative_path')[column]
                frame[column] = frame.relative_path.map(mapping).fillna(frame.get(column, ''))
    return normalize_manifest(frame)


def make_splits(frame, seed, test_size=.2, dev_size=.2):
    frame = normalize_manifest(frame)
    if frame.speaker_id.astype(str).str.lower().str.contains('unknown').any():
        raise ValueError('Isi speaker_id dari metadata terverifikasi sebelum group split.')
    for offset in range(100):
        trdev, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size,
                        random_state=seed+offset).split(frame, groups=frame.speaker_id))
        inner = frame.iloc[trdev]
        tr, va = next(GroupShuffleSplit(n_splits=1, test_size=dev_size,
                        random_state=seed+1000+offset).split(inner, groups=inner.speaker_id))
        parts = [inner.iloc[tr].copy(), inner.iloc[va].copy(), frame.iloc[te].copy()]
        if all(p.label.nunique() == 2 for p in parts):
            for p, name in zip(parts, ['train', 'dev', 'eval']):
                p['split'] = name
            result = pd.concat(parts, ignore_index=True)
            leakage_audit(result)
            return result
    raise ValueError('Tidak dapat membagi speaker menjadi tiga split dengan dua kelas; perbaiki metadata/populasi.')


def import_glottal(source_run, data_root, out):
    """Import only v9 selections, preserving IDs, labels and split; remap runtime paths."""
    source_run, data_root, out = Path(source_run), Path(data_root), Path(out)
    files = [p for p in sorted((source_run/'manifests').glob('*.csv')) if not p.name.startswith('_')]
    if not files:
        raise FileNotFoundError('Manifest run v9 tidak ditemukan.')
    frames, provenance = [], {}
    for path in files:
        frame = pd.read_csv(path, keep_default_na=False)
        frame = normalize_manifest(frame)
        remapped = []
        for row in frame.to_dict('records'):
            dataset, uid = row['dataset'].lower(), row['utterance_id']
            if dataset == 'asvspoof2019_la':
                p = data_root/'external/asvspoof2019_la'/f"ASVspoof2019_LA_{row['split']}"/'flac'/f'{uid}.flac'
            elif dataset == 'asvspoof2021_df':
                base = data_root/'external/asvspoof2021_df'
                choices = [base/'flac'/f'{uid}.flac', base/'ASVspoof2021_DF_eval/flac'/f'{uid}.flac', base/f'{uid}.flac']
                p = next((x for x in choices if x.exists()), choices[0])
            elif dataset.startswith('mlaad'):
                marker = '/data/external/mlaad_tiny/'
                if marker not in row['path']:
                    raise ValueError(f'Path MLAAD tidak dikenal: {row["path"]}')
                p = data_root/'external/mlaad_tiny'/row['path'].split(marker, 1)[1]
            elif dataset == 'additional':
                marker = '/ADDITIONAL/'
                if marker not in row['path']:
                    raise ValueError('Path ADDITIONAL tidak dikenal.')
                p = data_root/'sample/ADDITIONAL'/row['path'].split(marker, 1)[1]
            else:
                raise ValueError(f'Dataset v9 tidak didukung importer: {dataset}')
            remapped.append(str(p.resolve()))
        frame['source_path'] = frame.path
        frame['path'] = remapped
        frames.append(frame)
        provenance[path.name] = {'sha256': file_hash(path), 'rows': len(frame)}
    merged = normalize_manifest(pd.concat(frames, ignore_index=True))
    write_csv(out/'manifest.csv', merged)
    write_json(out/'source_v9.json', {'source_run': str(source_run), 'manifests': provenance,
                                    'note': 'Selection copied from glottal v9; features/models/results not reused.'})
    return merged


def read_audio(path):
    try:
        return sf.read(path, always_2d=False)
    except (RuntimeError, OSError):
        import librosa
        return librosa.load(path, sr=None, mono=True)
