"""Stable run identity and atomic artifact writes."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=str, allow_nan=False)
        name = stream.name
    os.replace(name, path)


def write_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        frame.to_csv(stream, index=False)
        name = stream.name
    os.replace(name, path)


def runtime_identity():
    names = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'librosa', 'soundfile',
             'xgboost', 'matplotlib', 'PyYAML', 'joblib']
    packages = {}
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = 'not-installed'
    return {'python': platform.python_version(), 'packages': packages}


def code_identity():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / 'src/pipeline_v9').glob('*.py')) + [root / 'scripts/run_pipeline_v9.py']
    return {str(p.relative_to(root)): file_hash(p) for p in paths if p.exists()}


FEATURE_CODE = ['audio_io.py', 'channels.py', 'common.py', 'extraction.py', 'features.py']
FEATURE_PACKAGES = ['numpy', 'scipy', 'librosa', 'soundfile', 'soxr', 'pandas']


def feature_code_identity():
    """Only modules that determine extracted values: training/report fixes keep the feature cache."""
    root = Path(__file__).resolve().parent
    return {name: file_hash(root / name) for name in FEATURE_CODE}


def feature_runtime_identity():
    packages = {}
    for name in FEATURE_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = 'not-installed'
    return {'python': platform.python_version(), 'packages': packages}


def copy_file(source, destination):
    """Write once to `destination` (e.g. Drive). No rename on the target filesystem: Drive's FUSE
    layer handles renames poorly, so readers treat a size mismatch as an incomplete copy."""
    import shutil
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if destination.stat().st_size != source.stat().st_size:
        raise OSError(f'Salinan tidak lengkap: {destination}')


def mirror_tree(source, destination):
    """Copy files missing (or differing in size) at destination. Never deletes."""
    source, destination = Path(source), Path(destination)
    copied = 0
    for path in sorted(source.rglob('*')):
        if path.is_file():
            target = destination / path.relative_to(source)
            if not target.is_file() or target.stat().st_size != path.stat().st_size:
                copy_file(path, target)
                copied += 1
    return copied
