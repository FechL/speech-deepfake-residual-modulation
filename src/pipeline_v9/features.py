"""Whole-utterance residual/modulation evidence and matched MFCC/LFCC baselines.

LPC residual is an all-pole prediction error, not a purified glottal source.
No family-specific duration/pitch gate. Invalid measurements remain NaN.
"""
from math import gcd
import warnings
import librosa
import numpy as np
from scipy.fft import dct
from scipy.signal import butter, hilbert, lfilter, resample_poly, sosfiltfilt
from scipy.stats import kurtosis

RESIDUAL_MEASURES = ['energy', 'entropy', 'prediction_error', 'irregularity', 'kurtosis']
MOD_NAMES = ['band_0p5_2', 'band_2_4', 'band_4_8', 'band_8_20', 'centroid_hz',
             'entropy_nats', 'envelope_variability']


def spectral_columns(kind, cfg):
    return [f'{kind}_{stat}_{i:02d}' for stat in ('static_mean', 'static_std', 'delta_std', 'delta2_std')
            for i in range(1, cfg['n_ceps'] + 1)]


def feature_families(cfg):
    residual = [f'res_{m}_{s}' for s in ('mean', 'std', 'median') for m in RESIDUAL_MEASURES]
    modulation = [f'mod_{m}' for m in MOD_NAMES]
    mfcc, lfcc = spectral_columns('mfcc', cfg), spectral_columns('lfcc', cfg)
    families = {'MFCC': mfcc, 'LFCC': lfcc, 'MFCC_LFCC': mfcc + lfcc,
                'Residual': residual, 'Modulation': modulation,
                'Residual_Modulation': residual + modulation,
                'QC_only': ['qc_original_seconds', 'qc_analysis_seconds', 'qc_peak',
                            'qc_rms', 'qc_active_ratio', 'qc_residual_valid_ratio']}
    for band in MOD_NAMES[:4]:
        families[f'Modulation_without_{band}'] = [c for c in modulation if c != f'mod_{band}']
    return families


def preprocess(raw, original_sr, cfg):
    x = np.asarray(raw, dtype=float)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Audio kosong, bentuk tidak valid, atau mengandung NaN/Inf.')
    sr = cfg['sample_rate']
    original_seconds = len(x) / original_sr
    peak = float(np.max(np.abs(x)))
    if original_sr != sr:
        x = librosa.resample(x, orig_sr=original_sr, target_sr=sr)
    max_seconds = cfg.get('max_seconds')
    crop_start = 0
    if max_seconds and len(x) > round(max_seconds * sr):
        n = round(max_seconds * sr)
        crop_start = (len(x) - n) // 2
        x = x[crop_start:crop_start + n]
    crop_seconds = len(x) / sr
    if cfg.get('trim_db') is not None and np.max(np.abs(x)) > 0:
        x, _ = librosa.effects.trim(x, top_db=cfg['trim_db'],
                                   frame_length=cfg['frame_length'], hop_length=cfg['hop_length'])
    if len(x) < round(cfg['min_seconds'] * sr):
        raise ValueError('Audio terlalu pendek setelah preprocessing bersama.')
    scale = float(np.max(np.abs(x)))
    if scale > 0:
        x = x / scale
    qc = {'qc_original_seconds': original_seconds, 'qc_analysis_seconds': len(x) / sr,
          'qc_crop_start_s': crop_start / sr, 'qc_crop_seconds': crop_seconds,
          'qc_peak': peak, 'qc_rms': float(np.sqrt(np.mean(x ** 2))),
          'qc_was_cropped': bool(max_seconds and original_seconds > max_seconds)}
    return x, sr, qc


def spectral_features(x, sr, cfg, active):
    """Same c1..cN, static mean/std, delta/std and delta2/std scheme as glottal v9."""
    frames = librosa.util.frame(x, frame_length=cfg['frame_length'], hop_length=cfg['hop_length']).T
    power = np.abs(np.fft.rfft(frames * np.hanning(cfg['frame_length']), n=cfg['n_fft'], axis=1)).T ** 2
    hz = np.fft.rfftfreq(cfg['n_fft'], 1 / sr)
    edges = np.linspace(0, sr / 2, cfg['n_filters'] + 2)
    linear = np.array([np.maximum(0, np.minimum((hz-a)/(b-a), (c-hz)/(c-b)))
                       for a, b, c in zip(edges[:-2], edges[1:-1], edges[2:])])
    mel = librosa.filters.mel(sr=sr, n_fft=cfg['n_fft'], n_mels=cfg['n_filters'])
    out, frame_cache = {}, {}
    for kind, bank in [('mfcc', mel), ('lfcc', linear)]:
        cep = dct(np.log(np.maximum(bank @ power, cfg['epsilon'])), axis=0, norm='ortho')[1:cfg['n_ceps']+1]
        if cfg['spectral_cmn']:
            cep -= cep.mean(axis=1, keepdims=True)
        width = min(9, cep.shape[1] - (1 - cep.shape[1] % 2))
        delta = librosa.feature.delta(cep, width=width, mode='nearest') if width >= 3 else np.zeros_like(cep)
        delta2 = librosa.feature.delta(cep, width=width, order=2, mode='nearest') if width >= 3 else np.zeros_like(cep)
        mat = np.vstack([cep, delta, delta2]).T[active]
        frame_cache[kind] = mat.astype(np.float32)
        n = cfg['n_ceps']
        vectors = [mat[:, :n].mean(0), mat[:, :n].std(0), mat[:, n:2*n].std(0), mat[:, 2*n:].std(0)] if len(mat) else [np.full(n, np.nan)] * 4
        out.update(zip(spectral_columns(kind, cfg), np.concatenate(vectors)))
    return out, frame_cache


def residual_frame(frame, cfg):
    windowed = frame * np.hamming(len(frame))
    signal_energy = float(np.mean(windowed ** 2))
    if signal_energy < cfg['residual_min_energy']:
        return None
    # Burg LPC, the reference estimator in panduan.pdf. Not a Wiener-Hopf solver.
    a = librosa.lpc(windowed.astype(float), order=cfg['lpc_order'])
    if not np.isfinite(a).all():
        raise ValueError('Koefisien LPC tidak finite.')
    residual = lfilter(a, [1.0], windowed)
    # Discard filter startup so the energy ratio uses identical support and window.
    residual = residual[cfg['lpc_order']:]
    support = windowed[cfg['lpc_order']:]
    energy = float(np.mean(residual ** 2))
    counts, _ = np.histogram(residual, bins=cfg['entropy_bins'])
    p = counts[counts > 0] / counts.sum()
    std = float(np.std(residual))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        k = float(kurtosis(residual, fisher=False, bias=False)) if std > cfg['epsilon'] else np.nan
    return [energy, float(-np.sum(p * np.log2(p))),
            energy / (np.mean(support ** 2) + cfg['epsilon']),
            float(np.mean(np.abs(np.diff(residual))) / (std + cfg['epsilon'])), k], residual


def residual_features(x, cfg):
    frames = librosa.util.frame(x, frame_length=cfg['frame_length'], hop_length=cfg['hop_length']).T
    rows, failed, low = [], 0, 0
    example = np.zeros(cfg['frame_length'] - cfg['lpc_order'])
    for frame in frames:
        try:
            result = residual_frame(frame, cfg)
            if result is None:
                low += 1
                continue
            values, residual = result
            if not np.isfinite(values[:4]).all():
                raise ValueError('Residual tidak finite.')
            rows.append(values)
            example = residual
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            failed += 1
    cols = feature_families(cfg)['Residual']
    if rows:
        arr = np.asarray(rows)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            values = np.concatenate([np.nanmean(arr, 0), np.nanstd(arr, 0), np.nanmedian(arr, 0)])
    else:
        values = np.full(len(cols), np.nan)
    return dict(zip(cols, values)), {'qc_residual_frames': len(frames), 'qc_residual_valid_frames': len(rows),
        'qc_residual_failed_frames': failed, 'qc_residual_low_energy_frames': low,
        'qc_residual_valid_ratio': len(rows) / max(len(frames), 1)}, example


def modulation_features(x, sr, cfg):
    env = np.abs(hilbert(x))
    sos = butter(cfg['envelope_filter_order'], cfg['envelope_lowpass_hz'], fs=sr, output='sos')
    env = sosfiltfilt(sos, env)
    g = gcd(sr, cfg['envelope_sr'])
    env = resample_poly(env, cfg['envelope_sr'] // g, sr // g)
    env -= env.mean()
    power = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
    freq = np.fft.rfftfreq(len(env), 1 / cfg['envelope_sr'])
    keep = (freq > 0) & (freq <= cfg['modulation_max_hz'])
    freq, power = freq[keep], power[keep]
    total = float(power.sum())
    cols = feature_families(cfg)['Modulation']
    if total <= cfg['epsilon']:
        return dict.fromkeys(cols, np.nan), freq, np.full(len(freq), np.nan)
    p = power / total
    bands = []
    for i, (low, high) in enumerate(cfg['modulation_bands']):
        select = (freq >= low) & ((freq <= high) if i == 3 else (freq < high))
        bands.append(float(p[select].sum()))
    nz = p[p > 0]
    values = bands + [float((freq * p).sum()), float(-np.sum(nz * np.log(nz))),
                      float(np.std(env) / (np.mean(np.abs(env)) + cfg['epsilon']))]
    return dict(zip(cols, values)), freq, p


def extract_audio(raw, original_sr, cfg, return_frames=False):
    x, sr, qc = preprocess(raw, original_sr, cfg)
    frames = librosa.util.frame(x, frame_length=cfg['frame_length'], hop_length=cfg['hop_length']).T
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    active = (rms > cfg['epsilon']) & (rms >= rms.max() * 10 ** (-cfg['active_db']/20))
    spectral, frame_cache = spectral_features(x, sr, cfg, active)
    residual, quality, _ = residual_features(x, cfg)
    modulation, freq, power = modulation_features(x, sr, cfg)
    out = {**spectral, **residual, **modulation, **qc, **quality,
           'qc_active_ratio': float(active.mean()), 'analysis_status': 'ok'}
    out['qc_missing_evidence'] = sum(not np.isfinite(v) for v in [*residual.values(), *modulation.values()])
    if return_frames:
        return out, {**frame_cache, 'mod_freq': freq, 'mod_power': power}
    return out
