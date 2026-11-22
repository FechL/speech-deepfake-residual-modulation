"""Controlled noise and MP3 roundtrip matching the glottal v9 stress design."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np
import soundfile as sf
from .common import digest


def mp3_available():
    exe = shutil.which('ffmpeg')
    if not exe:
        return False
    result = subprocess.run([exe, '-hide_banner', '-encoders'], capture_output=True, text=True)
    return result.returncode == 0 and 'libmp3lame' in result.stdout


def transform(raw, sr, cfg, uid):
    raw = np.asarray(raw, dtype=float)
    if raw.ndim == 2:
        raw = raw.mean(axis=1)
    metadata = {}
    if cfg.get('noise_snr_db') is not None:
        power = float(np.mean(raw ** 2))
        if power <= 0:
            raise ValueError('SNR tidak terdefinisi pada audio sunyi.')
        noise = np.random.default_rng(int(digest(uid)[:16], 16)).normal(size=len(raw))
        noise *= np.sqrt(power/(10 ** (cfg['noise_snr_db']/10) * np.mean(noise ** 2)))
        raw = raw + noise
        metadata['variant_actual_snr_db'] = float(10*np.log10(power/np.mean(noise ** 2)))
    if cfg.get('mp3_bitrate') is not None:
        bitrate = cfg['mp3_bitrate']
        if bitrate not in ['128k', '64k', '32k']:
            raise ValueError('Bitrate MP3 di luar protokol v9.')
        if not mp3_available():
            raise RuntimeError('ffmpeg dengan libmp3lame tidak tersedia.')
        with tempfile.TemporaryDirectory(prefix='resmod_v9_mp3_') as tmp:
            source, target = Path(tmp)/'in.wav', Path(tmp)/'out.mp3'
            sf.write(source, raw, sr, subtype='PCM_24')
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(source),
                '-c:a', 'libmp3lame', '-b:a', bitrate, str(target)], check=True, capture_output=True)
            pcm = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(target),
                '-f', 'f32le', '-ac', '1', '-ar', str(sr), 'pipe:1'], check=True, capture_output=True).stdout
        decoded = np.frombuffer(pcm, dtype='<f4').astype(float)
        difference = len(decoded)-len(raw)
        metadata['variant_length_difference_samples'] = difference
        raw = decoded[:len(raw)] if difference >= 0 else np.pad(decoded, (0, -difference))
    return raw, metadata
