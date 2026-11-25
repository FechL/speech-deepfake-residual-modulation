"""Immutable development bundles and target-only evaluation (extraction lives in extraction.py)."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from .common import code_identity, copy_file, digest, file_hash, mirror_tree, runtime_identity, write_csv, write_json
from .config import load_config, resolve_families
from .data import leakage_audit
from .extraction import extract_manifest, load_features  # noqa: F401  (public API)
from .evaluation import (consistency, eer, feature_statistics, metric_values, paired_intervals)
from .training import fit_systems, grouped_cv, score
from .baseline_gmm import fit_pair, score_pair


def mirror_exact(source, destination):
    """Like common.mirror_tree, but also replaces same-size files whose content differs. Run outputs
    can change in the last float digit between executions (RandomForest parallel summation), so a
    size-only comparison could leave a stale table next to a newer complete.json."""
    source, destination = Path(source), Path(destination)
    copied = 0
    for path in sorted(source.rglob('*')):
        if path.is_file():
            target = destination/path.relative_to(source)
            if (not target.is_file() or target.stat().st_size != path.stat().st_size
                    or file_hash(target) != file_hash(path)):
                copy_file(path, target)
                copied += 1
    return copied


def train_and_evaluate(feature_dir, config_path, results_root, checkpoints_root, allow_failures=False, backup=None):
    """`backup` (e.g. a Drive folder) receives runs/<id> and checkpoints/<id>: each frozen seed as soon
    as it exists, the result package at the end. A new session restores frozen seeds from it."""
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
    (Path(results_root)/'LATEST_RUN').write_text(run_id + '\n')
    restored = False
    if backup is not None:
        backup = Path(backup)
        for kind, local in [('runs', out), ('checkpoints', ckpt)]:
            if (backup/kind/run_id).is_dir():
                print(f'Memulihkan {mirror_tree(backup/kind/run_id, local)} file {kind} dari cadangan.', flush=True)
                restored = True
    if (out/'complete.json').exists():
        completed = json.loads((out/'complete.json').read_text())
        changed = [relative for relative, checksum in completed['artifacts_sha256'].items()
                   if not (out/relative).is_file() or file_hash(out/relative) != checksum]
        if changed and not restored:
            raise ValueError(f'Artifact run selesai berubah: {changed[0]}')
        if changed:
            # A backup mixing tables from two executions of the same run: frozen seeds are verified
            # separately (freeze.json), so evaluation is recomputed and the backup overwritten.
            print(f'Cadangan run tidak konsisten ({len(changed)} file, mis. {changed[0]}); '
                  'evaluasi dihitung ulang dari seed beku dan cadangan diperbaiki.', flush=True)
            (out/'complete.json').unlink()
        else:
            print(f'Run selesai sebelumnya: {out}', flush=True)
            if backup is not None:
                mirror_exact(out, backup/'runs'/run_id)
                mirror_exact(ckpt, backup/'checkpoints'/run_id)
            return out
    write_json(out/'run_identity.json', {'run_id': run_id, **identity, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'command': sys.argv, 'excluded_extraction_failures': n_failed,
        'scope': 'available_subset' if n_failed else 'planned', 'source_feature_dir': str(Path(feature_dir).resolve())})
    write_json(out/'leakage_audit.json', audit)
    write_csv(out/'manifest.csv', data[[c for c in data if c not in set(sum(resolve_families(cfg).values(), []))]])
    train, dev = data[data.split.eq('train')], data[data.split.eq('dev')]
    families = resolve_families(cfg)
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
            if backup is not None:
                mirror_exact(destination, backup/'checkpoints'/run_id/f'seed_{seed}')
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
            # Configured controls, e.g. does evidence add anything beyond the QC/duration shortcut?
            for family_a, family_b in cfg.get('paired_comparisons') or []:
                for clf in cfg.get('paired_classifiers') or [cfg['reference_classifier']]:
                    ka, kb = f'{family_a}__{clf}', f'{family_b}__{clf}'
                    pair = paired_intervals(frame, {ka: (wide[ka].to_numpy(), selected[ka]['threshold']),
                                                    kb: (wide[kb].to_numpy(), selected[kb]['threshold'])},
                                            cfg['bootstrap_draws'], seed, cfg['minimum_bootstrap_speakers'])
                    pair['scenario'], pair['seed'], pair['comparison'] = scenario, seed, f'{family_a} vs {family_b} ({clf})'
                    intervals_all.append(pair)
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
    if backup is not None:
        mirror_exact(ckpt, backup/'checkpoints'/run_id)
        mirror_exact(out, backup/'runs'/run_id)
        print(f'Cadangan hasil: {backup}', flush=True)
    print(f'Completed: {out}', flush=True)
    return out
