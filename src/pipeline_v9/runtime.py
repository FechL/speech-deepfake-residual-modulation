"""Resumable extraction, immutable development bundles, and target-only evaluation."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from .common import code_identity, digest, file_hash, runtime_identity, write_csv, write_json
from .config import load_config
from .data import leakage_audit, normalize_manifest, read_audio
from .features import extract_audio, feature_families
from .evaluation import (consistency, eer, feature_statistics, metric_values, paired_intervals)
from .training import fit_systems, grouped_cv, score
from .baseline_gmm import cache_path, fit_pair, score_pair


def extract_manifest(manifest_path, config_path, out, jobs=1, chunk=32):
    from joblib import Parallel, delayed
    cfg = load_config(config_path)
    manifest = normalize_manifest(pd.read_csv(manifest_path, keep_default_na=False))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Actual content hashes, including when reusing cache; no stale stat-only cache.
    actual_hashes = {}
    for row in manifest.itertuples():
        try:
            actual = file_hash(row.path)
        except OSError:
            actual = 'missing'
        if row.sha256 and actual != row.sha256:
            raise ValueError(f'Checksum input berubah/tidak tersedia: {row.utterance_id}')
        actual_hashes[row.utterance_id] = actual
    identity = {'features': cfg['features'], 'manifest_sha256': file_hash(manifest_path),
                'audio_sha256': actual_hashes, 'code': code_identity(), 'runtime': runtime_identity()}
    identity_hash = digest(identity)
    meta_path = out/'extraction_identity.json'
    if meta_path.exists() and json.loads(meta_path.read_text())['identity_hash'] != identity_hash:
        raise ValueError('Identitas cache berbeda. Gunakan folder output baru; cache lama tidak ditimpa.')
    write_json(meta_path, {'identity_hash': identity_hash, **identity})
    done = pd.read_csv(out/'features.csv') if (out/'features.csv').exists() else pd.DataFrame(columns=['utterance_id', 'analysis_status'])
    complete = {uid for uid in done.loc[done.analysis_status.eq('ok'), 'utterance_id'] if cache_path(out, uid).is_file()}
    todo = manifest[~manifest.utterance_id.isin(complete)]
    (out/'frames').mkdir(exist_ok=True)

    def one(row):
        base = {'utterance_id': row['utterance_id'], 'audio_sha256': actual_hashes[row['utterance_id']]}
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

    print(f'{len(complete)} cached; {len(todo)} to extract', flush=True)
    for start in range(0, len(todo), chunk):
        rows = Parallel(n_jobs=jobs)(delayed(one)(r) for r in todo.iloc[start:start+chunk].to_dict('records'))
        part = pd.DataFrame(rows)
        done = pd.concat([done[~done.utterance_id.isin(part.utterance_id)], part], ignore_index=True)
        write_csv(out/'features.csv', done)
        print(f'{min(start+chunk, len(todo))}/{len(todo)}', flush=True)
    if done.empty:
        raise ValueError('Manifest kosong.')
    joined = manifest.merge(done, on='utterance_id', how='left', validate='1:1')
    coverage = joined.groupby(['dataset', 'split', 'label'], dropna=False).agg(
        n=('utterance_id', 'size'), extracted=('analysis_status', lambda s: s.eq('ok').sum())).reset_index()
    coverage['coverage'] = coverage.extracted/coverage.n
    write_csv(out/'coverage.csv', coverage)
    write_csv(out/'failures.csv', joined[~joined.analysis_status.eq('ok')])
    manifest['sha256'] = manifest.utterance_id.map(actual_hashes).replace('missing', '')
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
    frame = manifest.merge(pd.read_csv(folder/'features.csv'), on='utterance_id', how='left', validate='1:1')
    invalid = ~frame.analysis_status.eq('ok')
    if invalid.any() and not allow_failures:
        raise ValueError(f'{invalid.sum()} ekstraksi gagal; periksa failures.csv. --allow-failures diperlukan untuk subset eksploratif.')
    if frame.empty:
        raise ValueError('Fitur kosong.')
    return frame[~invalid].reset_index(drop=True), identity, int(invalid.sum())


def train_and_evaluate(feature_dir, config_path, results_root, checkpoints_root, allow_failures=False):
    cfg = load_config(config_path)
    data, feature_identity, n_failed = load_features(feature_dir, allow_failures)
    if feature_identity['features'] != cfg['features']:
        raise ValueError('Konfigurasi fitur tidak sama dengan cache ekstraksi.')
    audit = leakage_audit(data)
    identity = {'config': cfg, 'feature_identity': feature_identity['identity_hash'],
                'code': code_identity(), 'runtime': runtime_identity(),
                'feature_table_sha256': file_hash(Path(feature_dir)/'features.csv'),
                'manifest_sha256': file_hash(Path(feature_dir)/'manifest.csv'),
                'allow_failures': allow_failures}
    run_id = digest(identity)[:16]
    out, ckpt = Path(results_root)/run_id, Path(checkpoints_root)/run_id
    out.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    if (out/'complete.json').exists():
        completed = json.loads((out/'complete.json').read_text())
        for relative, checksum in completed['artifacts_sha256'].items():
            if not (out/relative).is_file() or file_hash(out/relative) != checksum:
                raise ValueError(f'Artifact run selesai berubah: {relative}')
        print(f'Run selesai sebelumnya: {out}', flush=True)
        return out
    write_json(out/'run_identity.json', {'run_id': run_id, **identity, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'command': sys.argv, 'excluded_extraction_failures': n_failed,
        'scope': 'available_subset' if n_failed else 'planned', 'source_feature_dir': str(Path(feature_dir).resolve())})
    write_json(out/'leakage_audit.json', audit)
    write_csv(out/'manifest.csv', data[[c for c in data if c not in set(sum(feature_families(cfg['features']).values(), []))]])
    train, dev = data[data.split.eq('train')], data[data.split.eq('dev')]
    families = feature_families(cfg['features'])
    stats_columns = families['Residual_Modulation']
    write_csv(out/'statistical_tests.csv', feature_statistics(train, stats_columns, cfg, cfg['seeds'][0]))
    selections = []
    # Freeze every seed before evaluating any target.
    for seed in cfg['seeds']:
        destination = ckpt/f'seed_{seed}'
        destination.mkdir(exist_ok=True)
        bundle_path = destination/'models.joblib'
        if bundle_path.exists() and (destination/'freeze.json').exists():
            freeze = json.loads((destination/'freeze.json').read_text())
            if freeze['run_id'] != run_id or freeze['model_sha256'] != file_hash(bundle_path):
                raise ValueError('Bundle beku tidak cocok/rusak.')
            bundle = joblib.load(bundle_path)
        else:
            print(f'Training seed {seed}: {len(train)} train, {len(dev)} dev', flush=True)
            systems, champions, selection = fit_systems(train, dev, cfg, seed)
            gmms = {}
            if cfg['gmm']['enabled']:
                for kind in cfg['gmm']['kinds']:
                    models = fit_pair(train, feature_dir, kind, cfg['gmm'], seed)
                    scores = score_pair(models, dev, feature_dir, kind)
                    if not np.isfinite(scores).all():
                        raise ValueError('GMM dev memiliki utterance tanpa frame; jangan memilih threshold dari subset diam-diam.')
                    gmms[kind] = {'models': models, 'threshold': eer(dev.label, scores)[1]}
            bundle = dict(systems=systems, champions=champions, gmms=gmms,
                          selection=selection, seed=seed, run_id=run_id)
            temporary = bundle_path.with_suffix('.tmp')
            joblib.dump(bundle, temporary)
            temporary.replace(bundle_path)
            write_json(destination/'freeze.json', {'run_id': run_id, 'seed': seed,
                'model_sha256': file_hash(bundle_path), 'champions': champions,
                'threshold_source': 'dev_only', 'features': {k: v['columns'] for k, v in systems.items()}})
        selections.append(bundle['selection'])
    write_csv(out/'development_selection.csv', pd.concat(selections, ignore_index=True))
    write_csv(out/'grouped_cv.csv', grouped_cv(train, cfg, cfg['seeds'][0]))
    scores_all, metrics_all, intervals_all, cases_all, consistency_all = [], [], [], [], []
    for seed in cfg['seeds']:
        bundle = joblib.load(ckpt/f'seed_{seed}'/'models.joblib')
        for (dataset, split), frame in data[~data.split.eq('train')].groupby(['dataset', 'split'], sort=True):
            if split == 'unassigned':
                raise ValueError('Manifest masih memiliki split unassigned.')
            frame = frame.reset_index(drop=True)
            scenario = f'{dataset}__{split}'
            wide = frame.copy()
            selected = bundle['systems']
            for key, spec in selected.items():
                values = score(spec, frame)
                wide[key] = values
                part = frame[['utterance_id', 'speaker_id', 'attack_id', 'dataset', 'split', 'label']].copy()
                part['system'], part['seed'], part['score'], part['threshold'] = key, seed, values, spec['threshold']
                part['prediction'] = (values >= spec['threshold']).astype(int)
                scores_all.append(part)
                metrics_all.append(dict(seed=seed, scenario=scenario, dataset=dataset, split=split, system=key,
                    family=spec['family'], classifier=spec['classifier'], coverage=1.0,
                    **metric_values(frame.label, values, spec['threshold'])))
                if cfg['complete_case'] and spec['family'] == cfg['evidence_family']:
                    valid = frame[spec['columns']].notna().all(axis=1)
                    if valid.any():
                        metrics_all.append(dict(seed=seed, scenario=scenario, dataset=dataset, split=split,
                            system=key+'__complete_case', family=spec['family'], classifier=spec['classifier'],
                            coverage=float(valid.mean()), **metric_values(frame.loc[valid, 'label'], values[valid], spec['threshold'])))
            # Explicit GMM coverage, no imputation of an empty utterance frame sequence.
            for kind, spec in bundle['gmms'].items():
                values = score_pair(spec['models'], frame, feature_dir, kind)
                valid = np.isfinite(values)
                key = f'{kind.upper()}__gmm'
                part = frame.loc[valid, ['utterance_id', 'speaker_id', 'attack_id', 'dataset', 'split', 'label']].copy()
                part['system'], part['seed'], part['score'], part['threshold'] = key, seed, values[valid], spec['threshold']
                part['prediction'] = (values[valid] >= spec['threshold']).astype(int)
                scores_all.append(part)
                if valid.any():
                    metrics_all.append(dict(seed=seed, scenario=scenario, dataset=dataset, split=split,
                        system=key, family=kind.upper(), classifier='gmm', coverage=float(valid.mean()),
                        **metric_values(frame.loc[valid, 'label'], values[valid], spec['threshold'])))
            b = bundle['champions'][cfg['baseline_family']]
            e = bundle['champions'][cfg['evidence_family']]
            # Include same-classifier comparison alongside development champions.
            systems_ci = {key: (wide[key].to_numpy(), selected[key]['threshold']) for key in dict.fromkeys([
                b, e, f"{cfg['baseline_family']}__{cfg['reference_classifier']}",
                f"{cfg['evidence_family']}__{cfg['reference_classifier']}"])}
            ci = paired_intervals(frame, systems_ci, cfg['bootstrap_draws'], seed, cfg['minimum_bootstrap_speakers'])
            ci['scenario'], ci['seed'] = scenario, seed
            intervals_all.append(ci)
            cases, summary = consistency(frame, wide[b], wide[e], selected[b]['threshold'], selected[e]['threshold'])
            cases['seed'], cases['scenario'] = seed, scenario
            cases['baseline_system'], cases['evidence_system'] = b, e
            cases_all.append(cases)
            consistency_all.append(dict(seed=seed, scenario=scenario, baseline_system=b, evidence_system=e, **summary))
            # Attack panels retain bona fide references and never select models.
            for attack in sorted(frame.loc[frame.label.eq(1), 'attack_id'].unique()):
                mask = frame.label.eq(0) | frame.attack_id.eq(attack)
                for key in [b, e]:
                    metrics_all.append(dict(seed=seed, scenario=scenario, dataset=dataset, split=split,
                        system=key, panel_attack=attack, family=selected[key]['family'], classifier=selected[key]['classifier'],
                        coverage=1.0, **metric_values(frame.loc[mask, 'label'], wide.loc[mask, key], selected[key]['threshold'])))
    if not scores_all:
        raise ValueError('Tidak ada split evaluasi.')
    scores_frame, metrics_frame = pd.concat(scores_all, ignore_index=True), pd.DataFrame(metrics_all)
    cases = pd.concat(cases_all, ignore_index=True)
    write_csv(out/'utterance_scores.csv', scores_frame)
    write_csv(out/'metrics.csv', metrics_frame)
    write_csv(out/'bootstrap_intervals.csv', pd.concat(intervals_all, ignore_index=True))
    write_csv(out/'evidence_consistency.csv', pd.DataFrame(consistency_all))
    write_csv(out/'error_analysis.csv', cases)
    write_csv(out/'features.csv', data)
    write_csv(out/'measurement_coverage.csv', data.groupby(['dataset', 'split', 'label']).agg(
        n=('utterance_id', 'size'), residual_measured=('res_energy_mean', lambda s: s.notna().sum()),
        modulation_measured=('mod_centroid_hz', lambda s: s.notna().sum())).reset_index())
    group = ['scenario', 'system']
    pooled = metrics_frame[metrics_frame.get('panel_attack', pd.Series(np.nan, index=metrics_frame.index)).isna()]
    summary = pooled.groupby(group)[['auc', 'eer', 'f1_fake', 'balanced_accuracy', 'fpr', 'fnr']].agg(['mean', 'std']).reset_index()
    summary.columns = ['_'.join(filter(None, c)) if isinstance(c, tuple) else c for c in summary.columns]
    write_csv(out/'seed_summary.csv', summary)
    # Permutation reliance on the untouched target, with correlated groups permuted jointly.
    importance_rows = []
    seed0 = cfg['seeds'][0]
    reference = joblib.load(ckpt/f'seed_{seed0}'/'models.joblib')
    rng = np.random.default_rng(seed0)
    for (dataset, split), target in data[data.split.isin(['eval', 'external'])].groupby(['dataset', 'split']):
        if target.label.nunique() < 2:
            continue
        key = f"{cfg['evidence_family']}__{cfg['reference_classifier']}"
        spec = reference['systems'][key]
        original = roc_auc_score(target.label, score(spec, target))
        for group in ['Residual', 'Modulation']:
            columns = families[group]
            drops = []
            for _ in range(cfg['permutation_repeats']):
                altered = target.copy()
                altered[columns] = target[columns].to_numpy()[rng.permutation(len(target))]
                drops.append(original-roc_auc_score(target.label, score(spec, altered)))
            importance_rows.append(dict(dataset=dataset, split=split, system=key, group=group,
                auc_drop_mean=float(np.mean(drops)), auc_drop_sd=float(np.std(drops, ddof=1)) if len(drops)>1 else 0,
                note='Prediction reliance, not causal evidence'))
    write_csv(out/'grouped_permutation_importance.csv', pd.DataFrame(importance_rows))
    from .visuals import make_figures
    make_figures(out, data, scores_frame, cases, cfg, feature_dir)
    checksums = {str(p.relative_to(out)): file_hash(p) for p in out.rglob('*') if p.is_file() and p.name != 'complete.json'}
    write_json(out/'complete.json', {'run_id': run_id, 'status': 'complete', 'artifacts_sha256': checksums})
    print(f'Completed: {out}', flush=True)
    return out
