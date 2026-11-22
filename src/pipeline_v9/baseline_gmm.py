"""Matched MFCC/LFCC frame GMM reference; training frames only."""
from pathlib import Path
import numpy as np
from sklearn.mixture import GaussianMixture
from .common import digest


def cache_path(folder, uid):
    return Path(folder)/'frames'/f'{digest(uid)}.npz'


def fit_pair(train, cache_dir, kind, cfg, seed):
    rng = np.random.default_rng(seed)
    models = []
    for label in [0, 1]:
        part = train[train.label.eq(label)]
        part = part.sample(min(len(part), cfg['max_utterances_per_class']), random_state=seed)
        arrays = []
        # Reservoir with a deterministic per-file cap bounds memory by max_frames.
        per_file = max(cfg['n_components'], cfg['max_frames']//max(len(part), 1))
        for row in part.itertuples():
            with np.load(cache_path(cache_dir, row.utterance_id)) as frames:
                x = frames[kind]
            if len(x):
                arrays.append(x[rng.choice(len(x), min(len(x), per_file), replace=False)])
        if not arrays:
            raise ValueError(f'Tidak ada frame GMM kelas {label}.')
        x = np.concatenate(arrays)
        x = x[rng.choice(len(x), min(len(x), cfg['max_frames']), replace=False)]
        if len(x) < cfg['n_components']:
            raise ValueError('Frame training lebih sedikit daripada komponen GMM; gunakan config smoke.')
        model = GaussianMixture(n_components=cfg['n_components'], covariance_type='diag',
                                reg_covar=1e-5, random_state=seed).fit(x)
        models.append(model)
    return models


def score_pair(models, frame, cache_dir, kind):
    scores = []
    for row in frame.itertuples():
        with np.load(cache_path(cache_dir, row.utterance_id)) as frames:
            x = frames[kind]
        scores.append(float((models[1].score_samples(x)-models[0].score_samples(x)).mean()) if len(x) else np.nan)
    return np.array(scores)
