"""G1-G7 figures with explicit dataset/split panels and source artifacts."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from .baseline_gmm import cache_path
from .common import write_csv
from .data import read_audio
from .features import preprocess, residual_frame


def make_figures(out, data, scores, cases, cfg, feature_dir):
    folder = Path(out)/'figures'
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({'figure.dpi': 120, 'font.size': 10})

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(folder/f'{name}.png')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')
    for x, y, text in [(.07, .5, 'Audio + protocol\nwhole utterance'), (.38, .8, 'MFCC / LFCC\nclassification'),
                       (.38, .2, 'LPC residual + modulation\nevidence'), (.73, .5, 'Frozen train/dev systems\nutterance scores'),
                       (.96, .5, 'Consistency\n+ errors')]:
        ax.text(x, y, text, ha='center', va='center', bbox=dict(boxstyle='round', fc='#e5efff'), transform=ax.transAxes)
    for a, b in [((.16, .5), (.27, .8)), ((.16, .5), (.25, .2)), ((.5, .8), (.62, .5)),
                 ((.52, .2), (.62, .5)), ((.85, .5), (.9, .5))]:
        ax.annotate('', xy=b, xytext=a, xycoords='axes fraction', arrowprops={'arrowstyle': '->'})
    save(fig, 'G1_pipeline')
    # Acoustic examples come from train only; samples are a deterministic first 10/class.
    example_rows = []
    for label in [0, 1]:
        for number, row in enumerate(data[data.split.eq('train') & data.label.eq(label)].sort_values('utterance_id').head(10).itertuples()):
            raw, sr = read_audio(row.path)
            x, sr, _ = preprocess(raw, sr, cfg['features'])
            frame_length, hop = cfg['features']['frame_length'], cfg['features']['hop_length']
            starts = range(0, len(x)-frame_length+1, hop)
            start = max(starts, key=lambda s: float(np.mean(x[s:s+frame_length]**2)))
            frame = x[start:start+frame_length]
            result = residual_frame(frame, cfg['features'])
            if result is None:
                continue
            residual = result[1]
            fig, axes = plt.subplots(2, 1, figsize=(8, 4))
            axes[0].plot(np.arange(len(frame))/sr, frame)
            axes[0].set_title(f'{row.utterance_id} | label={label} | high-energy 25 ms frame')
            axes[1].plot(np.arange(len(residual))/sr, residual)
            axes[1].set(xlabel='Time (s)', ylabel='LPC residual')
            name = f'G2_label{label}_{number:02d}'
            save(fig, name)
            example_rows.append(dict(utterance_id=row.utterance_id, label=label, frame_start_s=start/sr, figure=name+'.png'))
    write_csv(folder/'G2_example_index.csv', pd.DataFrame(example_rows))
    first_seed = cfg['seeds'][0]
    for (dataset, split), frame in data.groupby(['dataset', 'split']):
        token = f'{dataset}__{split}'.replace('/', '_')
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        for ax, measure in zip(axes, ['energy', 'entropy', 'prediction_error', 'irregularity', 'kurtosis']):
            for label, name in [(0, 'REAL'), (1, 'FAKE')]:
                v = frame.loc[frame.label.eq(label), f'res_{measure}_mean'].dropna()
                if len(v):
                    ax.hist(v, bins=20, density=True, alpha=.5, label=name)
            ax.set_title(measure)
        axes[0].legend()
        fig.suptitle(f'Residual | {dataset}/{split}')
        save(fig, f'G3_{token}')
        # Common 0.5-20 Hz grid. An utterance only contributes where its own FFT resolution reaches
        # (first bin = 1/duration); below that it is NaN, so the 0.5-2 Hz band is shown from the
        # utterances long enough to resolve it instead of the whole plot starting at the shortest one.
        grid = np.linspace(.5, 20, 80)
        fig, ax = plt.subplots(figsize=(8, 4))
        table = []
        rng = np.random.default_rng(first_seed)
        for label, name in [(0, 'REAL'), (1, 'FAKE')]:
            mats, speakers = [], []
            for row in frame[frame.label.eq(label)].itertuples():
                with np.load(cache_path(feature_dir, row.utterance_id)) as archive:
                    freq, power = archive['mod_freq'], archive['mod_power']
                if len(freq) > 1 and np.isfinite(power).all():
                    # density per Hz, rather than power/bin (which depends on duration).
                    mats.append(np.interp(grid, freq, power/(freq[1]-freq[0]), left=np.nan, right=np.nan))
                    speakers.append(str(row.speaker_id))
            if not mats:
                continue
            mat, speakers = np.array(mats), np.array(speakers)
            n_valid = np.isfinite(mat).sum(0)
            with np.errstate(all='ignore'):
                mean = np.where(n_valid > 0, np.nansum(mat, 0) / np.maximum(n_valid, 1), np.nan)
            ax.plot(grid, mean, label=f'{name} (n={len(mat)})')
            unique = np.unique(speakers)
            supported = len(unique) >= cfg['minimum_bootstrap_speakers'] and all('unknown' not in s.lower() for s in unique)
            if supported:
                draws = []
                for _ in range(cfg['bootstrap_draws']):
                    indices = np.concatenate([np.flatnonzero(speakers == g) for g in rng.choice(unique, len(unique), replace=True)])
                    sample = mat[indices]
                    count = np.isfinite(sample).sum(0)
                    draws.append(np.where(count > 0, np.nansum(sample, 0) / np.maximum(count, 1), np.nan))
                low, high = np.nanpercentile(draws, [2.5, 97.5], axis=0)
                ax.fill_between(grid, low, high, alpha=.2)
            else:
                low = high = np.full_like(grid, np.nan)
            table.extend(dict(label=label, frequency_hz=f, mean_density=m, ci_low=l, ci_high=h, n_utterances=int(n),
                              ci_status='speaker_clustered' if supported else 'unsupported')
                         for f, m, l, h, n in zip(grid, mean, low, high, n_valid))
        ax.set(xlabel='Modulation frequency (Hz)', ylabel='Normalized power density (1/Hz)', title=f'{dataset}/{split}')
        if table:
            ax.legend()
        save(fig, f'G4_{token}')
        write_csv(folder/f'G4_{token}.csv', pd.DataFrame(table))
    for scenario, frame in cases[cases.seed.eq(first_seed)].groupby('scenario'):
        token = scenario.replace('/', '_')
        fig, ax = plt.subplots(figsize=(5, 4))
        if frame.label.nunique() == 2:
            for key in ['baseline_score', 'evidence_score']:
                fpr, tpr, _ = roc_curve(frame.label, frame[key])
                ax.plot(fpr, tpr, label=key)
            ax.plot([0, 1], [0, 1], '--', color='grey')
            ax.legend()
        ax.set(xlabel='FPR', ylabel='TPR', title=scenario)
        save(fig, f'G5_{token}')
        fig, ax = plt.subplots(figsize=(5, 4))
        for label in [0, 1]:
            p = frame[frame.label.eq(label)]
            ax.scatter(p.baseline_score, p.evidence_score, s=8, alpha=.5, label=f'label={label}')
        ax.axvline(frame.baseline_threshold.iloc[0], color='grey', linestyle='--')
        ax.axhline(frame.evidence_threshold.iloc[0], color='grey', linestyle='--')
        ax.set(xlabel='Baseline score', ylabel='Evidence score', title=scenario)
        ax.legend()
        save(fig, f'G6_{token}')
        fig, ax = plt.subplots(figsize=(8, 4))
        counts = frame.category.value_counts()
        ax.barh(counts.index, counts.values)
        ax.set(xlabel='Utterances', title=scenario)
        save(fig, f'G7_{token}')
