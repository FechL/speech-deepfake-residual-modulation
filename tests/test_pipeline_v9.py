from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.pipeline_v9.config import load_config
from src.pipeline_v9.data import leakage_audit, make_splits
from src.pipeline_v9.features import extract_audio, modulation_features, residual_frame, feature_families
from src.pipeline_v9.evaluation import eer, metric_values, fdr_bh, paired_intervals, consistency
from src.pipeline_v9.training import fit_systems, score

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return load_config(ROOT/'configs/residual_modulation_v9.yaml')


def test_modulation_localizes_known_envelope(cfg):
    sr = 16000
    t = np.arange(4*sr)/sr
    x = (1+.6*np.sin(2*np.pi*6*t))*np.sin(2*np.pi*250*t)
    feats, _, _ = modulation_features(x, sr, cfg['features'])
    assert feats['mod_band_4_8'] > .9
    assert abs(feats['mod_centroid_hz']-6) < .15
    assert sum(feats[f'mod_{k}'] for k in ['band_0p5_2', 'band_2_4', 'band_4_8', 'band_8_20']) <= 1.000001


def test_residual_reduces_predictable_energy_and_is_scale_invariant(cfg):
    rng = np.random.default_rng(10)
    x = np.sin(np.arange(400)*.09) + .01*rng.normal(size=400)
    a, _ = residual_frame(x, cfg['features'])
    b, _ = residual_frame(x*2, cfg['features'])
    assert a[2] < .1
    assert np.isclose(b[0], 4*a[0])
    assert np.isclose(a[2], b[2], rtol=1e-6)


def test_silence_keeps_missing_evidence_and_schema(cfg):
    row = extract_audio(np.zeros(16000), 16000, cfg['features'])
    assert np.isnan(row['res_energy_mean'])
    assert np.isnan(row['mod_centroid_hz'])
    assert row['qc_residual_valid_ratio'] == 0
    assert set(sum(feature_families(cfg['features']).values(), [])) <= set(row)


def test_stereo_resample_and_crop(cfg):
    rng = np.random.default_rng(1)
    f = dict(cfg['features'], max_seconds=1)
    row = extract_audio(rng.normal(size=(48000, 2)), 24000, f)
    assert row['qc_original_seconds'] == 2
    assert row['qc_analysis_seconds'] <= 1
    assert row['qc_was_cropped']


def manifest():
    return pd.DataFrame([dict(utterance_id=f'u{i}_{label}', path=f'/tmp/u{i}_{label}', label=label,
        speaker_id=f's{i}', attack_id='a', dataset='d', split='unassigned', sha256=f'hash{i}_{label}')
        for i in range(18) for label in [0, 1]])


def test_speaker_split_and_duplicate_audio_rejected():
    frame = make_splits(manifest(), 2026)
    assert leakage_audit(frame)['internal_speaker_disjoint']
    train = frame.index[frame.split.eq('train')][0]
    target = frame.index[frame.split.eq('eval')][0]
    frame.loc[target, 'sha256'] = frame.loc[train, 'sha256']
    with pytest.raises(ValueError, match='sha256'):
        leakage_audit(frame)


def test_unknown_speakers_not_invented():
    frame = manifest()
    frame['speaker_id'] = 'unknown'
    with pytest.raises(ValueError, match='speaker_id'):
        make_splits(frame, 1)


def test_threshold_finite_and_test_not_rethresholded():
    y = np.array([0, 0, 1, 1])
    _, threshold = eer(y, np.ones(4))
    assert np.isfinite(threshold)
    values = metric_values(y, np.array([.1, .2, .8, .9]), 2)
    assert values['recall_fake'] == 0
    assert values['auc'] == 1
    assert values['threshold'] == 2


def test_fdr_and_cluster_support_gate():
    q = fdr_bh([.01, .04, np.nan, .03])
    np.testing.assert_allclose(q[[0, 1, 3]], [.03, .04, .04])
    frame = pd.DataFrame({'label': [0, 1]*3, 'speaker_id': ['s1']*2+['s2']*2+['s3']*2})
    intervals = paired_intervals(frame, {'a': (np.array([.1, .9]*3), .5)}, 20, 1, 5)
    assert intervals.status.iloc[0] == 'unsupported_few_or_unknown_speakers'


def test_train_only_scaler_and_schema_no_metadata(cfg):
    rng = np.random.default_rng(5)
    columns = feature_families(cfg['features'])['Modulation']
    train = pd.DataFrame(rng.normal(size=(20, len(columns))), columns=columns)
    dev = pd.DataFrame(rng.normal(size=(10, len(columns)))+100, columns=columns)
    for frame, prefix in [(train, 'tr'), (dev, 'va')]:
        frame['label'] = [0, 1]*(len(frame)//2)
        frame['speaker_id'] = [f'{prefix}_{i}' for i in range(len(frame))]
    cfg.update(families=['Modulation'], models=['lr'])
    systems, _, _ = fit_systems(train, dev, cfg, 2026)
    spec = systems['Modulation__lr']
    np.testing.assert_allclose(spec['model']['scaler'].mean_, train[columns].mean())
    assert 'label' not in spec['columns'] and 'speaker_id' not in spec['columns']
    assert np.isfinite(score(spec, dev)).all()


def test_consistency_four_categories():
    frame = pd.DataFrame({'label': [0, 0, 1, 1]})
    cases, summary = consistency(frame, [.1, .1, .1, .1], [.1, .9, .9, .1], .5, .5)
    assert cases.category.nunique() == 4
    assert summary['agreement'] == .5


def test_cache_rejects_changed_config_and_tampered_table(tmp_path, cfg):
    import soundfile as sf
    import yaml
    from src.pipeline_v9.common import write_csv
    from src.pipeline_v9.runtime import extract_manifest, load_features
    rng = np.random.default_rng(55)
    path = tmp_path/'sample.wav'
    sf.write(path, rng.normal(0, .05, 16000), 16000)
    frame = pd.DataFrame([dict(utterance_id='sample', path=str(path), speaker_id='s',
        label=0, split='external', dataset='unit_test', attack_id='bonafide')])
    manifest_path, config_path = tmp_path/'manifest.csv', tmp_path/'config.yaml'
    write_csv(manifest_path, frame)
    config_path.write_text(yaml.safe_dump(cfg))
    folder = tmp_path/'features'
    extract_manifest(manifest_path, config_path, folder)
    first, _, _ = load_features(folder)
    extract_manifest(manifest_path, config_path, folder)
    assert len(load_features(folder)[0]) == 1
    # A lost NPZ frame cache is rebuilt without duplicating the feature row.
    next((folder/'frames').glob('*.npz')).unlink()
    extract_manifest(manifest_path, config_path, folder)
    assert len(list((folder/'frames').glob('*.npz'))) == 1
    cfg['features']['lpc_order'] = 12
    config_path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='Identitas cache'):
        extract_manifest(manifest_path, config_path, folder)
    with (folder/'features.csv').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='cache berubah'):
        load_features(folder)


def test_external_overlap_reported_but_development_audio_reuse_rejected():
    frame = make_splits(manifest(), 2026)
    row = frame[frame.split.eq('train')].iloc[0].copy()
    row['utterance_id'], row['path'], row['sha256'] = 'external_id', '/tmp/external_audio', 'external_hash'
    row['split'], row['dataset'] = 'external', 'external_dataset'
    frame = pd.concat([frame, row.to_frame().T], ignore_index=True)
    report = leakage_audit(frame)
    assert row['speaker_id'] in report['external_shared_development_speakers']
    frame.loc[frame.split.eq('external'), 'sha256'] = frame.loc[frame.split.eq('train'), 'sha256'].iloc[0]
    with pytest.raises(ValueError, match='sha256'):
        leakage_audit(frame)


def test_channel_noise_snr_and_silent_guard():
    from src.pipeline_v9.channels import transform
    t = np.arange(16000)/16000
    x = .3*np.sin(2*np.pi*200*t)
    y, info = transform(x, 16000, {'noise_snr_db': 20}, 'utterance1')
    assert abs(info['variant_actual_snr_db']-20) < 1e-8
    np.testing.assert_array_equal(y, transform(x, 16000, {'noise_snr_db': 20}, 'utterance1')[0])
    with pytest.raises(ValueError, match='sunyi'):
        transform(np.zeros(16000), 16000, {'noise_snr_db': 20}, 'silent')


def test_mp3_roundtrip_length_when_available():
    from src.pipeline_v9.channels import mp3_available, transform
    if not mp3_available():
        pytest.skip('ffmpeg/libmp3lame unavailable')
    t = np.arange(16000)/16000
    x = .2*np.sin(2*np.pi*250*t)
    y, info = transform(x, 16000, {'mp3_bitrate': '64k'}, 'codec')
    assert len(x) == len(y)
    assert np.isfinite(y).all()
    assert 'variant_length_difference_samples' in info


def test_preprocessing_follows_glottal_v9(cfg):
    from src.pipeline_v9.features import preprocess
    rng = np.random.default_rng(3)
    x = 0.2 * rng.normal(size=32000) + 0.1
    y, sr, qc = preprocess(x, 16000, cfg['features'])
    assert abs(y.mean()) < 1e-6 and abs(np.sqrt(np.mean(y ** 2)) - 1) < 1e-6
    assert qc['qc_rms'] > 0.1  # RMS before normalisation, not a constant


def test_audit_is_resumable_and_reports_missing(tmp_path, capsys):
    import soundfile as sf
    from src.pipeline_v9.data import audit_audio
    rows = []
    for i in range(3):
        path = tmp_path / f'a{i}.wav'
        sf.write(path, np.zeros(1600), 16000)
        rows.append(dict(utterance_id=f'u{i}', path=str(path), label=i % 2, speaker_id=f's{i}',
                         attack_id='a', split='external', dataset='d'))
    frame = pd.DataFrame(rows)
    out = tmp_path / 'audited.csv'
    first = audit_audio(frame, out, jobs=2, chunk=2)
    assert first.audio_status.eq('ready').all() and first.sha256.str.len().eq(64).all()
    capsys.readouterr()
    second = audit_audio(frame, out, jobs=2)
    assert '3 dari cache, 0 diperiksa' in capsys.readouterr().out
    assert second.sha256.tolist() == first.sha256.tolist()
    frame.loc[3] = dict(utterance_id='u3', path=str(tmp_path / 'missing.wav'), label=0, speaker_id='s3',
                        attack_id='a', split='external', dataset='d')
    with pytest.raises(FileNotFoundError, match='1/4'):
        audit_audio(frame, out)


def _wav_manifest(tmp_path, n=3, seconds=1.0):
    import soundfile as sf
    rng = np.random.default_rng(7)
    rows = []
    for i in range(n):
        path = tmp_path/'src'/f'u{i}.wav'
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, rng.normal(0, .05, int(16000*seconds)), 16000)
        rows.append(dict(utterance_id=f'u{i}', path=str(path), label=i % 2, speaker_id=f's{i}',
                         attack_id='a' if i % 2 else 'bonafide', split='external', dataset='d'))
    return pd.DataFrame(rows)


def test_stage_from_zip_and_files_hashes_written_bytes(tmp_path):
    import hashlib
    import zipfile
    from src.pipeline_v9.stage import stage_audio
    files = _wav_manifest(tmp_path, n=2)
    archive = tmp_path/'LA.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for uid in ['LA_T_1', 'LA_T_2', 'LA_T_3']:
            z.writestr(f'LA/ASVspoof2019_LA_train/flac/{uid}.flac', (tmp_path/'src'/'u0.wav').read_bytes())
    la = pd.DataFrame([dict(utterance_id=u, path='/drive/nope.flac', label=0, speaker_id='s', attack_id='bonafide',
                            split='train', dataset='la') for u in ['LA_T_1', 'LA_T_3']])
    manifest = tmp_path/'manifest.csv'
    pd.concat([files, la]).to_csv(manifest, index=False)
    out = tmp_path/'stage'
    staged = stage_audio(manifest, out, zips={'la': archive}, copy_zip=['la'])
    assert staged.utterance_id.tolist() == ['u0', 'u1', 'LA_T_1', 'LA_T_3']
    assert staged.audio_status.eq('ready').all()
    assert all(Path(p).is_relative_to(out.resolve()) for p in staged.path)
    for row in staged.itertuples():
        assert hashlib.sha256(Path(row.path).read_bytes()).hexdigest() == row.sha256
    assert sorted(p.name for p in (out/'audio'/'la').iterdir()) == ['LA_T_1.flac', 'LA_T_3.flac']
    assert not (out/'zips'/'LA.zip').exists()  # local archive copy removed after staging
    assert stage_audio(manifest, out, zips={'la': archive}).sha256.tolist() == staged.sha256.tolist()
    la.loc[1, 'utterance_id'] = 'LA_T_9'
    pd.concat([files, la]).to_csv(manifest, index=False)
    with pytest.raises(FileNotFoundError, match='1/4'):
        stage_audio(manifest, tmp_path/'stage2', zips={'la': archive})


def test_extraction_shards_backup_restore_and_path_independent_identity(tmp_path, cfg):
    import shutil
    import yaml
    from src.pipeline_v9.extraction import extract_manifest, load_features
    frame = _wav_manifest(tmp_path, n=3)
    manifest, config = tmp_path/'manifest.csv', tmp_path/'config.yaml'
    frame.to_csv(manifest, index=False)
    config.write_text(yaml.safe_dump(cfg))
    backup = tmp_path/'drive_backup'
    first = tmp_path/'session1'
    extract_manifest(manifest, config, first, chunk=2, backup=backup)
    shards = sorted((first/'shards').glob('*.csv'))
    assert len(shards) == 2
    [identity_dir] = list(backup.iterdir())
    assert sorted(p.name for p in (identity_dir/'shards').iterdir()) == sorted(
        [p.name for p in shards] + [p.with_suffix('.tar').name for p in shards])
    reference = load_features(first)[0].set_index('utterance_id')
    # New session: audio moved to another folder, empty local work dir; everything comes from the backup.
    moved = tmp_path/'elsewhere'
    shutil.copytree(tmp_path/'src', moved)
    frame['path'] = [str(moved/Path(p).name) for p in frame.path]
    frame.to_csv(manifest, index=False)
    for f in (tmp_path/'src').iterdir():
        f.unlink()
    second = tmp_path/'session2'
    extract_manifest(manifest, config, second, chunk=2, backup=backup)
    assert sorted(p.name for p in (second/'shards').glob('*.csv')) == [p.name for p in shards]  # nothing re-extracted
    restored = load_features(second)[0].set_index('utterance_id')
    assert np.allclose(restored['res_energy_mean'], reference['res_energy_mean'])
    assert len(list((second/'frames').glob('*.npz'))) == 3
    # A corrupt backup shard is skipped and its utterances are extracted again.
    tar = sorted((identity_dir/'shards').glob('*.tar'))[0]
    tar.write_bytes(b'not a tar')
    third = tmp_path/'session3'
    extract_manifest(manifest, config, third, chunk=2, backup=backup)
    assert len(load_features(third)[0]) == 3
