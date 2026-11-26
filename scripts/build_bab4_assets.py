"""Build the final Bab 4 tables and figures from three finished run packages (read-only).

Inputs are the verified run folders downloaded from the Drive backup:
  --main      main run (9 families, QC controls, analysis/)       e.g. backup/full/runs/1d5ba7b2b9562c28
  --stress    stress results of that run                          e.g. backup/full/stress/1d5ba7b2b9562c28/results
  --ablation  single-band ablation run                            e.g. backup/full/band_ablation/runs/52760a671e6853be
Every run's complete.json checksums are verified before anything is read. Outputs go to --out:
  tables/*.csv, tabel_bab4.md (all tables, Indonesian decimal comma), figures/*.png, sha256_sumber.txt.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

# Categorical slots 1-5 of the validated reference palette (fixed order, follows the entity).
COLORS = {'MFCC_LFCC': '#2a78d6', 'Residual_Modulation': '#eb6834', 'Residual': '#1baf7a',
          'Modulation': '#eda100', 'QC_only': '#e87ba4'}
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
SCEN = {'asvspoof2019_la__eval': 'LA eval', 'asvspoof2021_df__external': 'DF', 'mlaad_tiny__external': 'MLAAD'}
FAMILIES = ['MFCC', 'LFCC', 'MFCC_LFCC', 'Residual', 'Modulation', 'Residual_Modulation',
            'QC_only', 'QC_plus_Residual_Modulation', 'QC_plus_MFCC_LFCC']
STRESS_ORDER = ['lpc_12', 'lpc_20', 'trim_25db', 'trim_45db', 'no_trim', 'noise_snr30', 'noise_snr20',
                'noise_snr10', 'mp3_128k', 'mp3_64k', 'mp3_32k']
STRESS_LABEL = {'lpc_12': 'LPC orde 12', 'lpc_20': 'LPC orde 20', 'trim_25db': 'Trim 25 dB', 'trim_45db': 'Trim 45 dB',
                'no_trim': 'Tanpa trim', 'noise_snr30': 'Noise SNR 30 dB', 'noise_snr20': 'Noise SNR 20 dB*',
                'noise_snr10': 'Noise SNR 10 dB*', 'mp3_128k': 'MP3 128 kbps', 'mp3_64k': 'MP3 64 kbps',
                'mp3_32k': 'MP3 32 kbps'}
BANDS = ['0p5_2', '2_4', '4_8', '8_20']
BAND_LABEL = {'0p5_2': '0,5–2 Hz', '2_4': '2–4 Hz', '4_8': '4–8 Hz', '8_20': '8–20 Hz'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(run, allowed=()):
    """Checksums from complete.json. `allowed` names files known to differ (recorded, never silent)."""
    files = json.loads((run/'complete.json').read_text())['artifacts_sha256']
    bad = sorted(k for k, v in files.items() if not (run/k).is_file() or sha(run/k) != v)
    if set(bad) - set(allowed):
        raise ValueError(f'{run}: checksum tidak cocok {sorted(set(bad) - set(allowed))}')
    return {'files': len(files), 'known_mismatch': bad}


def num(x, digits=3):
    return '—' if pd.isna(x) else f'{x:.{digits}f}'.replace('.', ',').replace('-', '−')


def pct(x, digits=1):
    return num(100 * x, digits)


def ci(row, digits=3, signed=True):
    f = (lambda v: ('+' if v >= 0 else '−') + num(abs(v), digits)) if signed else (lambda v: num(v, digits))
    return f"{f(row['median'])} [{f(row['low'])}; {f(row['high'])}]"


def md_table(frame):
    head = '| ' + ' | '.join(map(str, frame.columns)) + ' |'
    rule = '| ' + ' | '.join('---' for _ in frame.columns) + ' |'
    body = ['| ' + ' | '.join(map(str, r)) + ' |' for r in frame.itertuples(index=False)]
    return '\n'.join([head, rule, *body])


class Book:
    def __init__(self, out):
        self.out, self.parts = out, []
        (out/'tables').mkdir(parents=True, exist_ok=True)

    def add(self, key, title, display, note='', raw=None):
        (raw if raw is not None else display).to_csv(self.out/'tables'/f'{key}.csv', index=False)
        self.parts.append(f'## {title}\n\n{md_table(display)}\n' + (f'\n{note}\n' if note else ''))

    def write(self, header):
        (self.out/'tabel_bab4.md').write_text(header + '\n\n' + '\n'.join(self.parts))


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis='x', color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'.replace('.', ',').replace('-', '−')))


def save(fig, path):
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- main run
def main_tables(book, run, figs):
    s = pd.read_csv(run/'seed_summary.csv')
    m = pd.read_csv(run/'metrics.csv')
    m = m[m.panel_attack.isna()]

    cov = pd.read_csv(run/'measurement_coverage.csv')
    sil = pd.read_csv(run/'analysis/silence_summary.csv')
    t = cov.merge(sil[['dataset', 'split', 'label', 'silence_median_s', 'analysed_median_s']], on=['dataset', 'split', 'label'])
    t['kelas'] = t.label.map({0: 'REAL', 1: 'FAKE'})
    disp = pd.DataFrame({'Dataset': t.dataset, 'Split': t.split, 'Kelas': t.kelas, 'n': t.n.map('{:,}'.format).str.replace(',', '.'),
                         'Residual terukur (%)': (t.residual_measured/t.n).map(pct),
                         'Modulasi terukur (%)': (t.modulation_measured/t.n).map(pct),
                         'Median durasi dianalisis (s)': t.analysed_median_s.map(lambda v: num(v, 2)),
                         'Median hening terpangkas (s)': t.silence_median_s.map(lambda v: num(v, 2))})
    book.add('t01_data_cakupan', 'Tabel 4.1 Komposisi data, cakupan pengukuran, dan durasi', disp,
             'Hening terpangkas = durasi asli − durasi setelah trim 35 dB. Semua 47.373 utterance terekstraksi (`analysis_status = ok`).', t)

    st = pd.read_csv(run/'statistical_tests.csv')
    disp = pd.DataFrame({'Fitur': st.feature, 'Median REAL': st.median_real.map(lambda v: f'{v:.4g}'.replace('.', ',')),
                         'Median FAKE': st.median_fake.map(lambda v: f'{v:.4g}'.replace('.', ',')),
                         'Rank-biserial (FAKE − REAL)': st.rank_biserial_fake_minus_real.map(lambda v: ('+' if v >= 0 else '−') + num(abs(v))),
                         'q (BH-FDR)': st.p_fdr.map(lambda v: '< 0,001' if v < 1e-3 else num(v))})
    book.add('t02_statistik_fitur', 'Tabel 4.2 Perbedaan fitur residual dan modulasi antara REAL dan FAKE (LA train)', disp,
             'Mann–Whitney U deskriptif pada LA train (2.580 REAL, 22.800 FAKE); utterance dalam satu speaker tidak independen, sehingga q bersifat deskriptif. Rank-biserial + berarti FAKE lebih tinggi.', st)

    rows = []
    for scen in SCEN:
        for fam in FAMILIES:
            for clf in ['lr', 'hgb', 'gmm']:
                key = f'{fam}__{clf}'
                r = s[(s.scenario == scen) & (s.system == key)]
                if r.empty:
                    continue
                r = r.iloc[0]
                rows.append(dict(scenario=scen, family=fam, classifier=clf, auc=r.auc_mean, auc_sd=r.auc_std,
                                 eer=r.eer_mean, ba=r.balanced_accuracy_mean, f1=r.f1_fake_mean, fpr=r.fpr_mean, fnr=r.fnr_mean))
    res = pd.DataFrame(rows)
    for i, scen in enumerate(SCEN, 3):
        r = res[res.scenario == scen]
        n = m[(m.scenario == scen)].iloc[0]
        disp = pd.DataFrame({'Keluarga': r.family, 'Classifier': r.classifier.str.upper(),
                             'AUC': [num(a) + (f' ± {num(sd)}' if sd > 5e-4 else '') for a, sd in zip(r.auc, r.auc_sd)],
                             'EER (%)': r.eer.map(pct), 'BA (%)': r.ba.map(pct), 'F1 FAKE (%)': r.f1.map(pct),
                             'FPR (%)': r.fpr.map(pct), 'FNR (%)': r.fnr.map(pct)})
        book.add(f't0{i}_hasil_{scen.split("__")[0]}', f'Tabel 4.{i} Hasil {SCEN[scen]} (n = {int(n.n):,}; {int(n.n_real):,} REAL)'.replace(',', '.'),
                 disp, 'Rerata ± SD tiga seed (SD ditampilkan bila > 0,0005; LR dan SVM deterministik). Threshold EER dari dev LA, dibekukan. '
                 'GMM 64 komponen tingkat frame. BA = balanced accuracy.', r)

    b = pd.read_csv(run/'bootstrap_intervals.csv')
    d = b[(b.status == 'paired_difference') & (b.metric == 'auc') & (b.seed == 2026) & b.scenario.isin(SCEN)].copy()
    d = d[d.scenario != 'mlaad_tiny__external']
    label = d.comparison.where(d.comparison.notna(), d.system.str.replace('-', ' vs ', regex=False))
    disp = pd.DataFrame({'Data': d.scenario.map(SCEN), 'Perbandingan (A − B)': label,
                         'ΔAUC [CI 95%]': d.apply(ci, axis=1), 'Speaker': d.n_speakers})
    book.add('t06_perbandingan_berpasangan', 'Tabel 4.6 Selisih AUC berpasangan (seed 2026)', disp,
             'Bootstrap persentil berklaster speaker, 1.000 draw, utterance identik. Pasangan tanpa label config adalah pasangan bawaan antara champion dev (MFCC_LFCC SVM-RBF, Residual_Modulation RF) dan classifier acuan LR. MLAAD tidak didukung (3 speaker).', d)

    p = pd.read_csv(run/'metrics.csv')
    p = p[(p.scenario == 'asvspoof2019_la__eval') & p.panel_attack.notna()]
    p = p.groupby(['panel_attack', 'system']).auc.mean().unstack()
    disp = pd.DataFrame({'Serangan': p.index, 'MFCC_LFCC SVM': p['MFCC_LFCC__svm_rbf'].map(num),
                         'Residual_Modulation RF': p['Residual_Modulation__rf'].map(num),
                         'Selisih (RM − baseline)': (p['Residual_Modulation__rf'] - p['MFCC_LFCC__svm_rbf']).map(lambda v: ('+' if v >= 0 else '−') + num(abs(v)))})
    book.add('t07_per_serangan', 'Tabel 4.7 AUC per serangan, LA eval (sistem champion dev)', disp,
             'Champion dipilih dari AUC dev: MFCC_LFCC SVM-RBF dan Residual_Modulation RF. Rerata tiga seed; tiap serangan dibandingkan dengan seluruh REAL eval.', p.reset_index())

    q = pd.read_csv(run/'analysis/qc_feature_auc.csv')
    q = q[(q.quantity == 'silence_seconds') & (q.split != 'train')]
    a = pd.read_csv(run/'analysis/stratified_auc_summary.csv')
    a = a[a.system.isin([f'{f}__{c}' for f in ['MFCC_LFCC', 'Residual_Modulation', 'Residual', 'Modulation', 'QC_only'] for c in ['lr', 'hgb']])]
    a = a[a.dataset.isin(['asvspoof2019_la', 'asvspoof2021_df']) & a.split.isin(['eval', 'external'])]
    disp = pd.DataFrame({'Data': a.dataset.map({'asvspoof2019_la': 'LA eval', 'asvspoof2021_df': 'DF'}), 'Sistem': a.system,
                         'AUC gabungan': a.pooled_auc_mean.map(num), 'AUC dalam desil hening': a.auc_within_silence_seconds_mean.map(num),
                         'AUC dalam desil durasi': a.auc_within_qc_analysis_seconds_mean.map(num)})
    silence_note = ('AUC durasi hening sendiri (diorientasi): ' + '; '.join(f"{r.dataset}/{r.split} {num(r.auc_oriented)}" for r in q.itertuples())
                    + '. REAL memiliki hening tepi lebih panjang. AUC dalam desil = rerata berbobot AUC di 10 bin kuantil; deskriptif, bin dari data target.')
    book.add('t08_kontrol_hening', 'Tabel 4.8 Kontrol jalan pintas hening: AUC gabungan vs di dalam strata', disp, silence_note, a)

    g = pd.read_csv(run/'grouped_permutation_importance.csv')
    disp = pd.DataFrame({'Data': g.dataset, 'Kelompok diacak': g.group, 'Penurunan AUC (rerata ± SD)': [f'{num(x)} ± {num(y)}' for x, y in zip(g.auc_drop_mean, g.auc_drop_sd)]})
    book.add('t09_permutation_importance', 'Tabel 4.9 Grouped permutation importance, Residual_Modulation LR seed 2026', disp,
             'Ketergantungan prediksi, bukan bukti kausal.', g)

    e = pd.read_csv(run/'evidence_consistency.csv')
    e = e[e.scenario.isin(SCEN)].groupby('scenario')[['n', 'spearman_rho', 'agreement', 'cohen_kappa', 'n_both_correct',
                                                       'n_baseline_correct_evidence_wrong', 'n_baseline_wrong_evidence_correct', 'n_both_wrong']].mean()
    disp = pd.DataFrame({'Data': e.index.map(SCEN), 'Spearman ρ': e.spearman_rho.map(num), 'Agreement (%)': e.agreement.map(pct),
                         'Kappa': e.cohen_kappa.map(num),
                         **{c: (e[k]/e.n).map(pct) for c, k in [('Keduanya benar (%)', 'n_both_correct'),
                                                                 ('Hanya baseline benar (%)', 'n_baseline_correct_evidence_wrong'),
                                                                 ('Hanya RM benar (%)', 'n_baseline_wrong_evidence_correct'),
                                                                 ('Keduanya salah (%)', 'n_both_wrong')]}})
    book.add('t10_konsistensi', 'Tabel 4.10 Konsistensi keputusan champion baseline (MFCC_LFCC SVM) dan bukti (RM RF)', disp,
             'Rerata tiga seed pada threshold beku.', e.reset_index())

    # Figure: AUC per family, LR and HGB, three datasets (small multiples, one axis each).
    fams = ['MFCC_LFCC', 'Residual_Modulation', 'Residual', 'Modulation', 'QC_only']
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    for ax, scen in zip(axes, SCEN):
        style(ax)
        for j, fam in enumerate(fams):
            for k, clf in enumerate(['lr', 'hgb']):
                r = res[(res.scenario == scen) & (res.family == fam) & (res.classifier == clf)]
                y = j + (k - .5) * .36
                v = float(r.auc.iloc[0])
                ax.barh(y, v - .5, left=.5, height=.32, color=COLORS[fam], alpha=1 if clf == 'lr' else .55,
                        edgecolor=SURFACE, linewidth=1)
                ax.text(max(v, .5) + .008, y, num(v), va='center', fontsize=6.5, color=INK2)
        ax.axvline(.5, color=INK2, linewidth=.8)
        ax.set_xlim(.0, 1.0)
        ax.set_title(SCEN[scen], fontsize=9, color=INK, loc='left')
        ax.set_xlabel('AUC (batang dari 0,5)', fontsize=8, color=INK2)
    axes[0].set_yticks(range(len(fams)), fams)
    axes[0].invert_yaxis()
    fig.text(.01, -.04, 'Batang pekat = LR, batang pudar = HGB. Rerata tiga seed. AUC < 0,5: arah skor terbalik.', fontsize=7, color=INK2)
    save(fig, figs/'gambar_4_1_auc_per_keluarga.png')

    # Figure: pooled vs within-silence AUC (LA eval and DF), LR systems.
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), sharey=True)
    for ax, ds, title in zip(axes, ['asvspoof2019_la', 'asvspoof2021_df'], ['LA eval', 'DF']):
        style(ax)
        for j, fam in enumerate(fams):
            r = a[(a.dataset == ds) & (a.system == f'{fam}__lr')].iloc[0]
            ax.plot([r.pooled_auc_mean, r.auc_within_silence_seconds_mean], [j, j], color=GRID, linewidth=2, zorder=1)
            ax.scatter(r.pooled_auc_mean, j, s=46, color=COLORS[fam], edgecolor=SURFACE, linewidth=2, zorder=3, marker='o')
            ax.scatter(r.auc_within_silence_seconds_mean, j, s=46, color=SURFACE, edgecolor=COLORS[fam], linewidth=2, zorder=3, marker='o')
        ax.axvline(.5, color=INK2, linewidth=.8, linestyle=':')
        ax.set_xlim(.4, 1.0)
        ax.set_title(title, fontsize=9, color=INK, loc='left')
        ax.set_xlabel('AUC (LR)', fontsize=8, color=INK2)
    axes[0].set_yticks(range(len(fams)), fams)
    axes[0].invert_yaxis()
    fig.text(.01, -.05, 'Titik penuh = AUC gabungan; titik kosong = AUC di dalam desil durasi hening (hening setara).', fontsize=7, color=INK2)
    save(fig, figs/'gambar_4_2_kontrol_hening.png')

    for name, target in [('G1_pipeline.png', 'gambar_4_0_alur_pipeline.png'),
                         ('G3_asvspoof2019_la__eval.png', 'gambar_4_3_distribusi_fitur_la_eval.png'),
                         ('G4_asvspoof2019_la__eval.png', 'gambar_4_4_spektrum_modulasi_la_eval.png'),
                         ('G5_asvspoof2019_la__eval.png', 'gambar_4_5_roc_la_eval.png'),
                         ('G5_asvspoof2021_df__external.png', 'gambar_4_6_roc_df.png'),
                         ('G7_asvspoof2019_la__eval.png', 'gambar_4_7_kategori_error_la_eval.png')]:
        shutil.copyfile(run/'figures'/name, figs/target)
    return res


# ---------------------------------------------------------------- stress
def stress_tables(book, stress, figs):
    proto = json.loads((stress/'protocol.json').read_text())
    status = pd.read_csv(stress/'variant_status.csv')
    if not status.status.eq('ok').all():
        raise ValueError('Ada varian stress yang gagal.')
    m = pd.read_csv(stress/'metrics.csv')
    p = pd.read_csv(stress/'variant_paired_intervals.csv')
    systems = ['MFCC_LFCC__lr', 'Residual_Modulation__lr', 'Residual__lr', 'Modulation__lr', 'QC_only__lr']
    d = p[(p.status == 'paired_difference') & (p.metric == 'auc') & p.system.isin(systems) & (p.dataset != 'mlaad_tiny')]
    ref = m[(m.variant == 'reference') & (m.dataset != 'mlaad_tiny')].set_index(['dataset', 'system']).auc
    for i, (ds, title) in enumerate([('asvspoof2019_la', 'LA eval'), ('asvspoof2021_df', 'DF')], 11):
        x = d[d.dataset == ds]
        table = x.pivot_table(index='variant', columns='system', values=['low', 'median', 'high'], aggfunc='first')
        rows = []
        for v in STRESS_ORDER:
            row = {'Varian': STRESS_LABEL[v]}
            for sname in systems[:3] + ['QC_only__lr']:
                row[sname.replace('__lr', '')] = ci({'median': table[('median', sname)][v], 'low': table[('low', sname)][v], 'high': table[('high', sname)][v]})
            rows.append(row)
        head = {'Varian': 'AUC referensi', **{s.replace('__lr', ''): num(ref[(ds, s)]) for s in systems[:3] + ['QC_only__lr']}}
        n = m[(m.dataset == ds)].iloc[0]
        book.add(f't{i}_stress_{ds}', f'Tabel 4.{i} ΔAUC kondisi uji terhadap referensi, {title} (subset n = {int(n.n)}; {int(n.n_real)} REAL), LR',
                 pd.DataFrame([head, *rows]),
                 f"Model seed {proto['seed']} dan threshold dev dibekukan; subset tetap, 1 utterance per sel speaker × serangan. "
                 'CI 95% bootstrap berpasangan berklaster speaker. *SNR 20/10 dB: noise ditambahkan sebelum trim, sehingga trim 35 dB tidak lagi memangkas hening '
                 '(durasi hening terpangkas jadi 0); hasilnya interaksi noise × trim, bukan uji ketahanan noise murni.', x)
    ba = m[(m.dataset == 'asvspoof2019_la') & m.system.isin(systems[:2] + ['QC_only__lr'])].pivot_table(index='variant', columns='system', values='balanced_accuracy')
    disp = pd.DataFrame({'Varian': ['Referensi'] + [STRESS_LABEL[v] for v in STRESS_ORDER],
                         **{s.replace('__lr', ''): [pct(ba.loc[v, s]) for v in ['reference', *STRESS_ORDER]] for s in ba.columns}})
    book.add('t13_stress_ba_la', 'Tabel 4.13 Balanced accuracy pada threshold beku, LA eval subset (LR, %)', disp, '', ba.reset_index())

    # Figure: ΔAUC with CI, MFCC_LFCC vs Residual_Modulation, LA and DF.
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, ds, title in zip(axes, ['asvspoof2019_la', 'asvspoof2021_df'], ['LA eval subset', 'DF subset']):
        style(ax)
        for k, (sname, fam) in enumerate([('MFCC_LFCC__lr', 'MFCC_LFCC'), ('Residual_Modulation__lr', 'Residual_Modulation')]):
            x = d[(d.dataset == ds) & (d.system == sname)].set_index('variant')
            for j, v in enumerate(STRESS_ORDER):
                y = j + (k - .5) * .34
                r = x.loc[v]
                ax.plot([r.low, r.high], [y, y], color=COLORS[fam], linewidth=2, solid_capstyle='round')
                ax.scatter(r['median'], y, s=40, color=COLORS[fam], edgecolor=SURFACE, linewidth=2, zorder=3,
                           label=fam if j == 0 else None)
        ax.axvline(0, color=INK2, linewidth=.8)
        for j, v in enumerate(STRESS_ORDER):
            if v in ('noise_snr20', 'noise_snr10'):
                ax.axhspan(j - .5, j + .5, color=GRID, alpha=.5, zorder=0)
        ax.set_title(title, fontsize=9, color=INK, loc='left')
        ax.set_xlabel('ΔAUC terhadap referensi [CI 95%]', fontsize=8, color=INK2)
    axes[0].set_yticks(range(len(STRESS_ORDER)), [STRESS_LABEL[v] for v in STRESS_ORDER])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc='upper right', ncol=2, bbox_to_anchor=(.98, 1.04))
    fig.text(.01, -.04, 'LR, model dan threshold beku. Baris berarsir (*): trim gagal memangkas hening, bukan uji noise murni.', fontsize=7, color=INK2)
    save(fig, figs/'gambar_4_8_stress_delta_auc.png')
    return proto


# ---------------------------------------------------------------- ablation
def ablation_tables(book, run, figs):
    s = pd.read_csv(run/'seed_summary.csv')
    fams = ['Modulation', 'Modulation_shape_only', 'Modulation_bands_only'] + [f'Modulation_band_{b}_only' for b in BANDS] + ['Residual_Modulation', 'MFCC_LFCC']
    label = {'Modulation': 'Modulasi lengkap (7)', 'Modulation_shape_only': 'Bentuk saja (centroid, entropy, variabilitas)',
             'Modulation_bands_only': 'Empat pita saja', 'Residual_Modulation': 'Residual + modulasi', 'MFCC_LFCC': 'MFCC + LFCC',
             **{f'Modulation_band_{b}_only': f'Pita {BAND_LABEL[b]} saja' for b in BANDS}}
    rows = []
    for f in fams:
        row = {'Model': label[f]}
        for scen in SCEN:
            for clf in ['lr', 'hgb']:
                r = s[(s.scenario == scen) & (s.system == f'{f}__{clf}')]
                row[f'{SCEN[scen]} {clf.upper()}'] = num(r.auc_mean.iloc[0])
        rows.append(row)
    book.add('t14_ablation_auc', 'Tabel 4.14 AUC ablation fitur modulasi (rerata tiga seed)', pd.DataFrame(rows),
             'Energi pita adalah proporsi yang berjumlah ≈ 1 (median 0,992), sehingga desain "tanpa satu pita" tidak dapat diidentifikasi; '
             'ablation memakai tiap pita sendirian. Model dilatih ulang dengan resep sama; fitur dari cache yang sama dengan run utama.')
    b = pd.read_csv(run/'bootstrap_intervals.csv')
    d = b[(b.status == 'paired_difference') & (b.metric == 'auc') & b.scenario.isin(['asvspoof2019_la__eval', 'asvspoof2021_df__external'])]
    d = d.groupby(['scenario', 'comparison'])[['low', 'median', 'high']].median().reset_index()
    disp = pd.DataFrame({'Data': d.scenario.map(SCEN), 'Perbandingan (A − B)': d.comparison, 'ΔAUC [CI 95%]': d.apply(ci, axis=1)})
    book.add('t15_ablation_berpasangan', 'Tabel 4.15 Selisih AUC berpasangan ablation modulasi', disp,
             'Median tiga seed dari batas bootstrap berklaster speaker (1.000 draw).', d)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    show = fams[:7]
    for ax, scen in zip(axes, ['asvspoof2019_la__eval', 'asvspoof2021_df__external']):
        style(ax)
        for j, f in enumerate(show):
            v = float(s[(s.scenario == scen) & (s.system == f'{f}__lr')].auc_mean.iloc[0])
            ax.barh(j, v - .5, left=.5, height=.6, color=COLORS['Modulation'] if j < 3 else '#f3c766', edgecolor=SURFACE, linewidth=1)
            ax.text(max(v, .5) + .004, j, num(v), va='center', fontsize=7, color=INK2)
        ax.axvline(.5, color=INK2, linewidth=.8)
        ax.set_xlim(.4, .7)
        ax.set_title(SCEN[scen], fontsize=9, color=INK, loc='left')
        ax.set_xlabel('AUC LR (batang dari 0,5)', fontsize=8, color=INK2)
    axes[0].set_yticks(range(len(show)), [label[f] for f in show])
    axes[0].invert_yaxis()
    save(fig, figs/'gambar_4_9_ablation_modulasi.png')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--main', required=True)
    ap.add_argument('--stress', required=True)
    ap.add_argument('--ablation', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--main-known-mismatch', nargs='*', default=[],
                    help='file run utama yang diketahui berbeda dari complete.json (dicatat di keluaran)')
    args = ap.parse_args()
    run, stress, abl, out = map(Path, [args.main, args.stress, args.ablation, args.out])
    checks = {run.name: verify(run, args.main_known_mismatch), abl.name: verify(abl)}
    figs = out/'figures'
    figs.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.titleweight': 'bold'})
    book = Book(out)
    main_tables(book, run, figs)
    proto = stress_tables(book, stress, figs)
    ablation_tables(book, abl, figs)
    book.write(f'# Tabel Bab 4 — residual-modulasi v9\n\nSumber: run utama `{run.name}`, stress `{stress.parent.name}` '
               f'(subset {proto["subset_sha256"][:12]}…), ablation `{abl.name}`. Checksum `complete.json` terverifikasi: {checks}. '
               'Desimal memakai koma. Angka dibangkitkan oleh `scripts/build_bab4_assets.py`; jangan diedit manual.')
    sources = sorted([*run.glob('*.csv'), *run.glob('*.json'), *(run/'analysis').glob('*'), *stress.glob('*.csv'), *stress.glob('*.json'),
                      *abl.glob('*.csv'), *abl.glob('*.json')])
    (out/'sha256_sumber.txt').write_text(''.join(f'{sha(p)}  {p}\n' for p in sources))
    print(f'Selesai: {out}')


if __name__ == '__main__':
    main()
