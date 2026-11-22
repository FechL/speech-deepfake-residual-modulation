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
