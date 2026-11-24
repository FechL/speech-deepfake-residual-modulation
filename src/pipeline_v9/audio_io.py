"""Audio decoding shared by extraction, staging and figures."""
import soundfile as sf


def read_audio(path):
    try:
        return sf.read(path, always_2d=False)
    except (RuntimeError, OSError):
        import librosa
        return librosa.load(path, sr=None, mono=True)
