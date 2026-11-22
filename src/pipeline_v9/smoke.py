"""Synthetic integration fixture; never research evidence."""
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from .common import write_csv, write_json
from .runtime import extract_manifest, train_and_evaluate


def smoke(out):
    root = Path(__file__).resolve().parents[2]
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(101)
    rows = []
    sr = 16000
    # 5 test speakers support the minimum cluster bootstrap gate.
    for split, speakers in [('train', range(8)), ('dev', range(8, 13)), ('eval', range(13, 18)), ('external', range(18, 20))]:
        for speaker in speakers:
            for label in [0, 1]:
                t = np.arange(sr * 2)/sr
                f0 = 95 + 7*speaker
                x = (1+.4*np.sin(2*np.pi*(3+label*4)*t))*np.sin(2*np.pi*f0*t)
                x += (.03+.015*label)*rng.normal(size=len(t))
                uid = f'synthetic_{split}_{speaker}_{label}'
                path = out/'audio'/f'{uid}.wav'
                path.parent.mkdir(exist_ok=True)
                sf.write(path, .4*x, sr)
                rows.append(dict(utterance_id=uid, speaker_id=f'synthetic_{speaker}', label=label,
                    attack_id='synthetic_fake' if label else 'bonafide', split=split, dataset='SYNTHETIC_SMOKE', path=str(path)))
    manifest = out/'manifest.csv'
    write_csv(manifest, pd.DataFrame(rows))
    cfg = yaml.safe_load((root/'configs/residual_modulation_v9.yaml').read_text())
    cfg.update(seeds=[2026, 2027, 2028], models=['lr', 'svm_rbf', 'rf', 'hgb', 'xgb'],
               bootstrap_draws=12, permutation_repeats=2, group_cv_folds=2, n_jobs=1)
    cfg['features'].update(n_ceps=6, n_filters=12)
    cfg['gmm'].update(n_components=2, max_frames=2000, max_utterances_per_class=4)
    cfg['model_params']['hgb']['max_iter'] = 5
    cfg['model_params']['rf']['n_estimators'] = 5
    cfg['model_params']['xgb']['n_estimators'] = 5
    config = out/'smoke.yaml'
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    write_json(out/'SYNTHETIC_ONLY.json', {'warning': 'Generated signals for software validation; not dataset research results.'})
    extract_manifest(manifest, config, out/'features', jobs=1)
    return train_and_evaluate(out/'features', config, out/'runs', out/'checkpoints')
