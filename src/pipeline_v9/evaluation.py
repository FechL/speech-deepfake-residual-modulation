"""Frozen operating points and paired speaker-cluster uncertainty."""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    cohen_kappa_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score, roc_curve)


def eer(y, scores):
    fpr, tpr, thresholds = roc_curve(y, scores, drop_intermediate=False)
    fnr = 1-tpr
    # Exclude the +inf sentinel from selecting a deployable validation threshold.
    finite = np.flatnonzero(np.isfinite(thresholds))
    i = finite[np.argmin(np.abs(fpr[finite]-fnr[finite]))]
    return float((fpr[i]+fnr[i])/2), float(thresholds[i])


def metric_values(y, scores, threshold):
    y, scores = np.asarray(y, dtype=int), np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    out = {'n': len(y), 'n_real': int((y == 0).sum()), 'n_fake': int((y == 1).sum()),
        'threshold': float(threshold), 'accuracy': accuracy_score(y, pred),
        'precision_fake': precision_score(y, pred, zero_division=0),
        'recall_fake': recall_score(y, pred, zero_division=0),
        'recall_real': recall_score(y, pred, pos_label=0, zero_division=0),
        'f1_fake': f1_score(y, pred, zero_division=0),
        'f1_macro': f1_score(y, pred, average='macro', labels=[0, 1], zero_division=0),
        'mcc': matthews_corrcoef(y, pred)}
    out.update(fpr=1-out['recall_real'], fnr=1-out['recall_fake'])
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, scores)
        out.update(auc=roc_auc_score(y, scores), eer=eer(y, scores)[0],
            balanced_accuracy=balanced_accuracy_score(y, pred),
            average_precision=average_precision_score(y, scores),
            tpr_at_fpr_1pct=float(tpr[fpr <= .01].max()), tpr_at_fpr_5pct=float(tpr[fpr <= .05].max()))
    else:
        out.update(auc=np.nan, eer=np.nan, balanced_accuracy=np.nan, average_precision=np.nan,
                   tpr_at_fpr_1pct=np.nan, tpr_at_fpr_5pct=np.nan)
    return out


def cluster_indices(groups, draws, seed):
    groups = np.asarray(groups).astype(str)
    unique = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        yield np.concatenate([index[g] for g in rng.choice(unique, len(unique), replace=True)])


def paired_intervals(frame, systems, draws, seed, min_speakers):
    """All systems and deltas use the very same speaker resamples and utterances."""
    groups = frame.speaker_id.astype(str)
    known = ~groups.str.lower().str.contains('unknown')
    if not known.all() or groups.nunique() < min_speakers:
        return pd.DataFrame([dict(status='unsupported_few_or_unknown_speakers', n_speakers=groups.nunique())])
    values = {key: [] for key in systems}
    for idx in cluster_indices(groups, draws, seed):
        y = frame.label.to_numpy()[idx]
        if len(np.unique(y)) < 2:
            continue
        for key, (scores, threshold) in systems.items():
            m = metric_values(y, np.asarray(scores)[idx], threshold)
            values[key].append([m['auc'], m['eer'], m['f1_fake']])
    rows = []
    if not any(values.values()):
        return pd.DataFrame([{'status': 'unsupported_resamples'}])
    for a, samples in values.items():
        for j, metric in enumerate(['auc', 'eer', 'f1_fake']):
            lo, mid, hi = np.percentile(np.array(samples)[:, j], [2.5, 50, 97.5])
            rows.append(dict(system=a, metric=metric, low=lo, median=mid, high=hi,
                             status='ok', n_speakers=groups.nunique(), draws=len(samples)))
    names = list(systems)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            diff = np.asarray(values[a])-np.asarray(values[b])
            for j, metric in enumerate(['auc', 'eer', 'f1_fake']):
                lo, mid, hi = np.percentile(diff[:, j], [2.5, 50, 97.5])
                rows.append(dict(system=f'{a}-{b}', metric=metric, low=lo, median=mid, high=hi,
                                 status='paired_difference', n_speakers=groups.nunique(), draws=len(diff)))
    return pd.DataFrame(rows)


def fdr_bh(p):
    p = np.asarray(p, dtype=float)
    result = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    if len(order):
        q = p[order] * len(order)/np.arange(1, len(order)+1)
        result[order] = np.minimum(1, np.minimum.accumulate(q[::-1])[::-1])
    return result


def feature_statistics(frame, columns, cfg, seed):
    rows = []
    support = frame.speaker_id.nunique() >= cfg['minimum_bootstrap_speakers'] and not frame.speaker_id.astype(str).str.lower().str.contains('unknown').any()
    for c in columns:
        real = frame.loc[frame.label.eq(0), c].dropna()
        fake = frame.loc[frame.label.eq(1), c].dropna()
        row = dict(feature=c, n_real=len(real), n_fake=len(fake), median_real=real.median(), median_fake=fake.median())
        if len(real) and len(fake):
            u, p = mannwhitneyu(real, fake, alternative='two-sided')
            row.update(u=u, p=p, rank_biserial_fake_minus_real=1-2*u/(len(real)*len(fake)),
                       median_difference=float(fake.median()-real.median()))
        else:
            row.update(u=np.nan, p=np.nan, rank_biserial_fake_minus_real=np.nan, median_difference=np.nan)
        differences = []
        if support:
            for idx in cluster_indices(frame.speaker_id, cfg['bootstrap_draws'], seed):
                part = frame.iloc[idx]
                a = part.loc[part.label.eq(0), c].dropna()
                b = part.loc[part.label.eq(1), c].dropna()
                if len(a) and len(b):
                    differences.append(b.median()-a.median())
        lo, hi = np.percentile(differences, [2.5, 97.5]) if differences else (np.nan, np.nan)
        row.update(median_ci_low=lo, median_ci_high=hi, ci_status='speaker_clustered' if differences else 'unsupported',
                   test_note='MWU descriptive: utterances within speakers may be correlated')
        rows.append(row)
    result = pd.DataFrame(rows)
    result['p_fdr'] = fdr_bh(result.p)
    return result


def consistency(frame, baseline, evidence, baseline_threshold, evidence_threshold):
    out = frame.copy()
    b, e = np.asarray(baseline), np.asarray(evidence)
    bp, ep = b >= baseline_threshold, e >= evidence_threshold
    correct_b, correct_e = bp == out.label.to_numpy(), ep == out.label.to_numpy()
    out['baseline_score'], out['evidence_score'] = b, e
    out['baseline_threshold'], out['evidence_threshold'] = baseline_threshold, evidence_threshold
    out['baseline_pred'], out['evidence_pred'] = bp.astype(int), ep.astype(int)
    out['category'] = np.select([correct_b & correct_e, correct_b & ~correct_e, ~correct_b & correct_e],
        ['both_correct', 'baseline_correct_evidence_wrong', 'baseline_wrong_evidence_correct'], default='both_wrong')
    rho, p = spearmanr(b, e) if np.std(b) > 0 and np.std(e) > 0 else (np.nan, np.nan)
    kappa = cohen_kappa_score(bp, ep) if len(np.unique(np.r_[bp, ep])) > 1 else np.nan
    summary = dict(n=len(out), spearman_rho=rho, spearman_p=p,
                   agreement=float(np.mean(bp == ep)), cohen_kappa=kappa)
    summary.update({f'n_{k}': int(out.category.eq(k).sum()) for k in ['both_correct', 'baseline_correct_evidence_wrong',
                   'baseline_wrong_evidence_correct', 'both_wrong']})
    return out, summary
