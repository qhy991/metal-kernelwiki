#!/usr/bin/env python3
"""Install a thin metal-kernelwiki skill pointing to this checkout; never overwrite."""
import argparse
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
    if not (root / 'data/catalog.json').is_file() or not (source / 'SKILL.md').is_file():
        parser.error('Incomplete source repository')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns('knowledge','__pycache__','*.pyc'))
    (destination / 'knowledge').symlink_to(root, target_is_directory=True)
    print(destination)
    print('knowledge -> ' + str(root))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
