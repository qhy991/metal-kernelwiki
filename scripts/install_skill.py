#!/usr/bin/env python3
"""Install a thin metal-kernelwiki skill pointing to this checkout; never overwrite."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True

def main():
    root = Path(__file__).resolve().parents[1]
    codex_root = Path(os.environ.get('CODEX_HOME') or Path.home() / '.codex')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest', type=Path, default=codex_root / 'skills' / 'metal-kernelwiki')
    args = parser.parse_args()
    destination = args.dest.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        parser.error('Destination already exists; refusing to overwrite: ' + str(destination))
    if destination.name != 'metal-kernelwiki':
        parser.error('Destination directory must be named metal-kernelwiki')
    source = root / 'skill' / 'metal-kernelwiki'
    knowledge_entries = ('data', 'wiki', 'references', 'README.md', 'MAINTENANCE.md', 'PROVENANCE.md',
                         'README.en.md', 'MAINTENANCE.en.md', 'PROVENANCE.en.md')
    if (not (root / 'mwiki').is_file()
            or not (root / 'data/catalog.json').is_file()
            or not (source / 'SKILL.md').is_file()
            or any(not (root / name).exists() for name in knowledge_entries)):
        parser.error('Incomplete source repository')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns('knowledge','repository.json','__pycache__','*.pyc'))
    # Expose the knowledge, without exposing a second SKILL.md through repo/skill.
    knowledge = destination / 'knowledge'
    knowledge.mkdir()
    for name in knowledge_entries:
        target = root / name
        (knowledge / name).symlink_to(target, target_is_directory=target.is_dir())
    (destination / 'repository.json').write_text(
        json.dumps({'repository': str(root)}, ensure_ascii=False) + '\n', encoding='utf-8')
    print(destination)
    print('Knowledge links and repository.json point to ' + str(root))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
