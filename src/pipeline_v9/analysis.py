"""Post-hoc shortcut analysis of a finished run (no retraining; reads the run package only).

Why: in the first full runs QC_only reached AUC 0.875 on ASVspoof 2019 LA eval. The driver is
the leading/trailing silence removed by trimming (original minus analysed duration), a known
ASVspoof 2019 artefact. Features are computed on trimmed audio, yet a system can still pick up
correlates of silence. This module reports:

* qc_feature_auc.csv          univariate AUC of each QC quantity and of the silence duration;
* silence_summary.csv         silence/duration per dataset, split and label;
* stratified_auc.csv          per system and seed: pooled AUC and AUC within quantile bins of a
                              stratifier (silence seconds, analysed seconds), weighted by bin size;
* stratified_auc_summary.csv  the same averaged over seeds.

Within-bin AUC compares real and fake utterances with similar silence (or duration), so a system
whose pooled AUC comes from that quantity drops towards 0.5. It is descriptive: bins come from the
target data itself and no interval is attached. Higher score = more likely fake (label 1).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from .common import write_csv, write_json

QC = ['qc_original_seconds', 'qc_analysis_seconds', 'qc_peak', 'qc_rms', 'qc_active_ratio', 'qc_residual_valid_ratio']
STRATIFIERS = {'silence_seconds': 'Durasi hening yang dipangkas trim (asli - dianalisis)',
               'qc_analysis_seconds': 'Durasi audio setelah trim'}


def add_silence(frame):
    frame = frame.copy()
    frame['silence_seconds'] = frame.qc_original_seconds - frame.qc_analysis_seconds
    frame['silence_fraction'] = frame.silence_seconds / frame.qc_original_seconds
    return frame


def _auc(y, x):
    return float(roc_auc_score(y, x)) if len(np.unique(y)) == 2 else np.nan


def stratified_auc(labels, scores, strata, bins=10, min_per_class=5):
    """Weighted mean of AUC within quantile bins of `strata`. Bins lacking either class are skipped."""
    labels, scores, strata = np.asarray(labels), np.asarray(scores, dtype=float), np.asarray(strata, dtype=float)
    edges = np.unique(np.quantile(strata, np.linspace(0, 1, bins + 1)))
    index = np.digitize(strata, edges[1:-1])
    aucs, weights = [], []
    for b in np.unique(index):
        keep = index == b
        y = labels[keep]
        if min(int((y == 0).sum()), int((y == 1).sum())) >= min_per_class:
            aucs.append(roc_auc_score(y, scores[keep]))
            weights.append(int(keep.sum()))
    if not aucs:
        return np.nan, 0, 0
    return float(np.average(aucs, weights=weights)), len(aucs), int(sum(weights))


def analyze_run(run_dir, bins=10, out=None):
    run_dir = Path(run_dir)
    out = Path(out) if out else run_dir/'analysis'
    features = add_silence(pd.read_csv(run_dir/'features.csv', low_memory=False,
                                       usecols=lambda c: c in {'utterance_id', 'dataset', 'split', 'label', *QC}))
    quantities = QC + ['silence_seconds', 'silence_fraction']
    rows = []
    for (dataset, split), group in features.groupby(['dataset', 'split']):
        for column in quantities:
            auc = _auc(group.label, group[column])
            rows.append(dict(dataset=dataset, split=split, quantity=column, auc=auc,
                             auc_oriented=max(auc, 1 - auc) if np.isfinite(auc) else np.nan,
                             direction='fake lebih tinggi' if auc >= .5 else 'real lebih tinggi'))
    write_csv(out/'qc_feature_auc.csv', pd.DataFrame(rows))
    summary = (features.groupby(['dataset', 'split', 'label'])
               .agg(n=('utterance_id', 'size'), silence_median_s=('silence_seconds', 'median'),
                    silence_mean_s=('silence_seconds', 'mean'), analysed_median_s=('qc_analysis_seconds', 'median'),
                    original_median_s=('qc_original_seconds', 'median')).reset_index())
    write_csv(out/'silence_summary.csv', summary)

    scores = pd.read_csv(run_dir/'utterance_scores.csv',
                         usecols=['utterance_id', 'system', 'seed', 'dataset', 'split', 'label', 'score'])
    scores = scores.merge(features[['utterance_id', *STRATIFIERS]], on='utterance_id', how='left', validate='m:1')
    if scores[list(STRATIFIERS)].isna().any().any():
        raise ValueError('Skor tanpa QC/durasi; features.csv dan utterance_scores.csv tidak sepasang.')
    rows = []
    for (dataset, split, system, seed), group in scores.groupby(['dataset', 'split', 'system', 'seed'], sort=True):
        if split == 'train':
            continue
        row = dict(dataset=dataset, split=split, system=system, seed=seed, n=len(group),
                   pooled_auc=_auc(group.label, group.score))
        for stratifier in STRATIFIERS:
            value, used, covered = stratified_auc(group.label, group.score, group[stratifier], bins)
            row.update({f'auc_within_{stratifier}': value, f'bins_{stratifier}': used,
                        f'coverage_{stratifier}': covered / len(group)})
        rows.append(row)
    table = pd.DataFrame(rows)
    write_csv(out/'stratified_auc.csv', table)
    value_columns = ['pooled_auc', *[f'auc_within_{s}' for s in STRATIFIERS]]
    mean = table.groupby(['dataset', 'split', 'system'])[value_columns].agg(['mean', 'std']).reset_index()
    mean.columns = ['_'.join(filter(None, c)) for c in mean.columns]
    write_csv(out/'stratified_auc_summary.csv', mean)
    write_json(out/'analysis_protocol.json', {
        'run': run_dir.name, 'bins': bins, 'min_per_class_per_bin': 5, 'stratifiers': STRATIFIERS,
        'note': 'Deskriptif; bin dari data target; tanpa interval. Tidak dipakai untuk memilih model atau threshold.'})
    return out
