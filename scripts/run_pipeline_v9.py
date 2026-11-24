#!/usr/bin/env python3
"""Standalone residual/modulation v9 CLI. Run --help for the available stages."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('import-glottal-v9', help='Reuse exact v9 sample IDs and splits; remap data paths.')
    q.add_argument('--source-run', required=True)
    q.add_argument('--data-root', required=True)
    q.add_argument('--out', required=True)
    q = sub.add_parser('stage', help='Copy selected audio to local disk (from zip or files), hash bytes, audit leakage.')
    q.add_argument('--manifest', required=True)
    q.add_argument('--out', required=True, help='Local folder; writes audio/ and staged_manifest.csv.')
    q.add_argument('--zip', action='append', default=[], metavar='DATASET=PATH',
                   help='Read this dataset from a zip archive (by <utterance_id>.flac). Repeatable.')
    q.add_argument('--copy-zip', action='append', default=[], metavar='DATASET',
                   help='Copy that archive to local disk before reading (faster for large selections).')
    q.add_argument('--jobs', type=int, default=8)
    q.add_argument('--allow-missing', action='store_true')
    q = sub.add_parser('manifest-la')
    q.add_argument('--root', required=True)
    q.add_argument('--out', required=True)
    q = sub.add_parser('manifest-folder')
    q.add_argument('--root', required=True)
    q.add_argument('--dataset', required=True)
    q.add_argument('--split', choices=['external', 'unassigned'], default='external')
    q.add_argument('--metadata', help='CSV with relative_path and verified speaker_id, optional attack_id/source_id.')
    q.add_argument('--out', required=True)
    q = sub.add_parser('split')
    q.add_argument('--manifest', required=True)
    q.add_argument('--seed', type=int, default=2026)
    q.add_argument('--out', required=True)
    q = sub.add_parser('audit')
    q.add_argument('--manifest', required=True)
    q.add_argument('--out', required=True)
    q.add_argument('--allow-missing', action='store_true', help='Write audit without stopping for unavailable audio.')
    q.add_argument('--jobs', type=int, default=8, help='Parallel file reads (threads); I/O bound on Drive.')
    q.add_argument('--chunk', type=int, default=500, help='Checkpoint the audit cache every N files.')
    q = sub.add_parser('extract')
    q.add_argument('--manifest', required=True)
    q.add_argument('--config', default=str(ROOT/'configs/residual_modulation_v9.yaml'))
    q.add_argument('--out', required=True)
    q.add_argument('--jobs', type=int, default=1)
    q.add_argument('--chunk', type=int, default=500, help='Utterances per immutable shard.')
    q.add_argument('--backup', help='Write-once shard backup folder (e.g. on Drive); restored on resume.')
    q = sub.add_parser('run')
    q.add_argument('--features', required=True)
    q.add_argument('--config', default=str(ROOT/'configs/residual_modulation_v9.yaml'))
    q.add_argument('--out', default=str(ROOT/'results/residual_modulation_v9'))
    q.add_argument('--checkpoints', default=str(ROOT/'checkpoints/residual_modulation_v9'))
    q.add_argument('--allow-failures', action='store_true')
    q.add_argument('--backup', help='Mirror frozen seeds and the finished run here (e.g. on Drive).')
    q = sub.add_parser('stress', help='Frozen models on a fixed target subset; no retraining.')
    q.add_argument('--run', required=True)
    q.add_argument('--features', required=True)
    q.add_argument('--checkpoints', default=str(ROOT/'checkpoints/residual_modulation_v9'))
    q.add_argument('--per-cell', type=int, default=2)
    q.add_argument('--jobs', type=int, default=1)
    q = sub.add_parser('smoke', help='Generate explicitly synthetic signals and run the full pipeline.')
    q.add_argument('--out', default=str(ROOT/'results/smoke'))
    args = p.parse_args()
    import pandas as pd
    from src.pipeline_v9.common import write_csv, write_json
    from src.pipeline_v9.data import (import_glottal, manifest_la, manifest_folder, make_splits,
                                      normalize_manifest, leakage_audit, audit_audio)
    if args.command == 'import-glottal-v9':
        frame = import_glottal(args.source_run, args.data_root, args.out)
        existence = frame.copy()
        existence['file_exists'] = existence.path.map(lambda p: Path(p).is_file())
        write_csv(Path(args.out)/'path_availability.csv', existence)
        print(frame.groupby(['dataset', 'split', 'label']).size().to_string())
        print(f'Audio tersedia: {existence.file_exists.sum()}/{len(frame)}. Lihat path_availability.csv.')
    elif args.command == 'stage':
        from src.pipeline_v9.stage import stage_audio
        zips = {}
        for item in args.zip:
            name, sep, path = item.partition('=')
            if not sep or not name or not path:
                p.error('--zip harus berbentuk DATASET=PATH')
            zips[name] = path
        staged = stage_audio(args.manifest, args.out, zips, args.copy_zip, args.jobs, args.allow_missing)
        status = leakage_audit(staged, require_development=staged.split.eq('train').any())
        write_json(Path(args.out)/'staged_manifest.json', status)
        print(json.dumps(status, indent=2))
    elif args.command == 'manifest-la':
        write_csv(args.out, manifest_la(args.root))
    elif args.command == 'manifest-folder':
        write_csv(args.out, manifest_folder(args.root, args.dataset, args.split, args.metadata))
    elif args.command == 'split':
        write_csv(args.out, make_splits(pd.read_csv(args.manifest), args.seed))
    elif args.command == 'audit':
        frame = normalize_manifest(pd.read_csv(args.manifest, keep_default_na=False))
        audited = audit_audio(frame, args.out, args.allow_missing, args.jobs, args.chunk)
        status = leakage_audit(audited, require_development=frame.split.eq('train').any())
        write_json(Path(args.out).with_suffix('.json'), status)
        print(json.dumps(status, indent=2))
    elif args.command == 'extract':
        if args.chunk < 1 or args.jobs == 0:
            p.error('chunk harus positif dan jobs tidak boleh nol.')
        from src.pipeline_v9.extraction import extract_manifest
        extract_manifest(args.manifest, args.config, args.out, args.jobs, args.chunk, args.backup)
    elif args.command == 'run':
        from src.pipeline_v9.runtime import train_and_evaluate
        train_and_evaluate(args.features, args.config, args.out, args.checkpoints, args.allow_failures, args.backup)
    elif args.command == 'stress':
        from src.pipeline_v9.stress import run_stress
        run_stress(args.run, args.features, args.checkpoints, args.per_cell, args.jobs)
    elif args.command == 'smoke':
        from src.pipeline_v9.smoke import smoke
        smoke(args.out)


if __name__ == '__main__':
    main()
