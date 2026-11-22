"""Fixed-target, one-factor sensitivity and channel stress; frozen models only."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml
from .common import digest, file_hash, write_csv, write_json
from .evaluation import metric_values, paired_intervals
from .features import feature_families
from .channels import mp3_available
from .runtime import extract_manifest, load_features
from .training import score


def run_stress(run, feature_dir, checkpoints_root, per_cell=2, jobs=1):
    if per_cell < 1:
        raise ValueError('per-cell harus positif.')
    run = Path(run)
    identity = json.loads((run/'run_identity.json').read_text())
    cfg, run_id = identity['config'], identity['run_id']
    data, feature_identity, _ = load_features(feature_dir)
    if feature_identity['identity_hash'] != identity['feature_identity']:
        raise ValueError('Cache stress berbeda dari cache run utama.')
    seed = cfg['seeds'][0]
    bundle_path = Path(checkpoints_root)/run_id/f'seed_{seed}'/'models.joblib'
    freeze = json.loads(bundle_path.with_name('freeze.json').read_text())
    if freeze['run_id'] != run_id or freeze['model_sha256'] != file_hash(bundle_path):
        raise ValueError('Model beku stress tidak cocok/rusak.')
    bundle = joblib.load(bundle_path)
    targets = data[data.split.isin(['eval', 'external'])]
    if targets.empty:
        raise ValueError('Tidak ada target eval/external.')
    subset = pd.concat([g.sample(min(len(g), per_cell), random_state=seed)
        for _, g in targets.groupby(['dataset', 'split', 'speaker_id', 'label', 'attack_id'])], ignore_index=True)
    root = run/'stress'
    root.mkdir(exist_ok=True)
    write_csv(root/'subset_manifest.csv', subset)
    write_json(root/'protocol.json', {'seed': seed, 'per_cell': per_cell,
        'subset_sha256': file_hash(root/'subset_manifest.csv'), 'model_sha256': file_hash(bundle_path),
        'note': 'Fixed subset; frozen seed-0 models; GMM excluded. No retraining or target threshold selection.'})
    keys = [f'{family}__{cfg["reference_classifier"]}' for family in [cfg['baseline_family'], cfg['evidence_family'], 'Residual', 'Modulation', 'QC_only'] if f'{family}__{cfg["reference_classifier"]}' in bundle['systems']]
    rows, deltas, changes, statuses = [], [], [], []
    reference = None
    for variant in [{'name': 'reference'}, *cfg['sensitivity_variants']]:
        name = variant['name']
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            raise ValueError('Nama varian tidak aman.')
        if variant.get('mp3_bitrate') and not mp3_available():
            statuses.append(dict(variant=name, status='unsupported_ffmpeg_libmp3lame'))
            write_csv(root/'variant_status.csv', pd.DataFrame(statuses))
            continue
        settings = json.loads(json.dumps(cfg))
        for key, value in variant.items():
            if key != 'name':
                settings['features']['noise_snr_db' if key == 'snr_db' else key] = value
        folder = root/name
        folder.mkdir(exist_ok=True)
        config_path = folder/'config.yaml'
        config_path.write_text(yaml.safe_dump(settings, sort_keys=False))
        # Source is the manifest schema, never a joined feature table.
        manifest = subset[[c for c in ['utterance_id', 'path', 'label', 'speaker_id', 'attack_id', 'split', 'dataset', 'sha256', 'source_id'] if c in subset]]
        write_csv(folder/'manifest.csv', manifest)
        extracted = extract_manifest(folder/'manifest.csv', config_path, folder/'features', jobs=jobs)
        good = extracted.analysis_status.eq('ok')
        if not good.all():
            raise ValueError(f'Stress {name} gagal pada {int((~good).sum())} utterance; lihat failures.csv.')
        extracted = extracted.sort_values('utterance_id').reset_index(drop=True)
        if reference is None:
            reference = extracted.copy()
        if list(extracted.utterance_id) != list(reference.utterance_id):
            raise ValueError('Pasangan stress berubah.')
        statuses.append(dict(variant=name, status='ok', n=len(extracted)))
        for (dataset, split), frame in extracted.groupby(['dataset', 'split']):
            clean = reference[reference.utterance_id.isin(frame.utterance_id)].reset_index(drop=True)
            frame = frame.reset_index(drop=True)
            for key in keys:
                spec = bundle['systems'][key]
                values = score(spec, frame)
                rows.append(dict(variant=name, system=key, dataset=dataset, split=split,
                                 seed=seed, **metric_values(frame.label, values, spec['threshold'])))
                reference_values = score(spec, clean)
                ci = paired_intervals(frame, {'variant': (values, spec['threshold']),
                    'reference': (reference_values, spec['threshold'])}, cfg['bootstrap_draws'], seed,
                    cfg['minimum_bootstrap_speakers'])
                ci['variant'], ci['system'], ci['dataset'], ci['split'] = name, key, dataset, split
                deltas.append(ci)
            for column in feature_families(cfg['features'])['Residual_Modulation']:
                valid = frame[column].notna() & clean[column].notna()
                difference = frame.loc[valid, column]-clean.loc[valid, column]
                changes.append(dict(variant=name, dataset=dataset, split=split, feature=column,
                    paired_valid=int(valid.sum()), median_change=float(difference.median()) if len(difference) else np.nan,
                    missing_reference=float(clean[column].isna().mean()), missing_variant=float(frame[column].isna().mean())))
        write_csv(root/'metrics.csv', pd.DataFrame(rows))
        write_csv(root/'variant_status.csv', pd.DataFrame(statuses))
        write_csv(root/'variant_feature_change.csv', pd.DataFrame(changes))
        write_csv(root/'variant_paired_intervals.csv', pd.concat(deltas, ignore_index=True))
    print(f'Stress selesai: {root}')
