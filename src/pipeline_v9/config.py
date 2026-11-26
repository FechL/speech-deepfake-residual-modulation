from pathlib import Path
import yaml
from .features import feature_families


def resolve_families(cfg):
    """Base families from features.py, `column_families` (explicit columns) and `composite_families`
    (unions of families) from config.

    Both live in config, not in features.py, so adding a control model never changes the
    feature-extraction code and therefore never invalidates an extracted feature cache.
    """
    families = feature_families(cfg['features'])
    known = {c for columns in families.values() for c in columns}
    # column_families: explicit column subsets (e.g. one modulation band alone) for ablations.
    for name, columns in (cfg.get('column_families') or {}).items():
        if name in families or not columns or not set(columns) <= known or len(set(columns)) != len(columns):
            raise ValueError(f'column_families {name} tidak valid.')
        families[name] = list(columns)
    for name, parts in (cfg.get('composite_families') or {}).items():
        if name in families or not parts or not set(parts) <= set(families):
            raise ValueError(f'composite_families {name} tidak valid.')
        families[name] = list(dict.fromkeys(c for part in parts for c in families[part]))
    return families


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    f = cfg['features']
    if not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds']):
        raise ValueError('Seed kosong atau duplikat.')
    if cfg['reference_classifier'] not in cfg['models']:
        raise ValueError('reference_classifier harus tersedia pada models.')
    families = resolve_families(cfg)
    if not set(cfg['families']) <= set(families):
        raise ValueError('Keluarga fitur tidak dikenal.')
    for pair in cfg.get('paired_comparisons') or []:
        if len(pair) != 2 or not set(pair) <= set(cfg['families']):
            raise ValueError(f'paired_comparisons {pair} harus dua family yang dilatih.')
    if not set(cfg.get('paired_classifiers') or [cfg['reference_classifier']]) <= set(cfg['models']):
        raise ValueError('paired_classifiers harus tersedia pada models.')
    if not {cfg['baseline_family'], cfg['evidence_family']} <= set(cfg['families']):
        raise ValueError('Baseline/evidence family wajib tersedia.')
    if f['sample_rate'] != 16000 or f['frame_length'] != 400 or f['hop_length'] != 160:
        raise ValueError('Pipeline ini menggunakan 16 kHz, frame 25 ms, dan hop 10 ms.')
    if not 1 <= f['lpc_order'] < f['frame_length']-2:
        raise ValueError('Orde LPC tidak valid.')
    if not 0 < f['n_ceps'] < f['n_filters'] or f['n_fft'] < f['frame_length']:
        raise ValueError('Konfigurasi spektral tidak valid.')
    if f['min_seconds'] < f['frame_length']/f['sample_rate']:
        raise ValueError('Minimum durasi kurang dari satu frame.')
    if not 0 < f['modulation_max_hz'] < f['envelope_lowpass_hz'] < f['envelope_sr']/2:
        raise ValueError('Cutoff envelope/frekuensi modulasi tidak valid.')
    if f['modulation_bands'] != [[.5, 2], [2, 4], [4, 8], [8, 20]]:
        raise ValueError('Nama fitur memerlukan empat pita modulasi yang ditetapkan panduan.')
    if cfg['selection_metric'] != 'auc':
        raise ValueError('Seleksi v9 menggunakan AUC development.')
    if cfg['bootstrap_draws'] < 1 or cfg['minimum_bootstrap_speakers'] < 2:
        raise ValueError('Konfigurasi bootstrap tidak valid.')
    return cfg
