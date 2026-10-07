#!/usr/bin/env python3
"""Find the canonical library in an installed skill or repository template."""
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True

def main():
    skill = Path(__file__).resolve().parents[1]
    for root in (skill / 'knowledge', skill.parent.parent):
        launcher = root / 'mwiki'
        if launcher.is_file() and (root / 'data/catalog.json').is_file():
            return subprocess.call([sys.executable, str(launcher), *sys.argv[1:]])
    print('Metal KernelWiki knowledge link is missing; install from the repository.', file=sys.stderr)
    return 1

if __name__ == '__main__':
    raise SystemExit(main())
