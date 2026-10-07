#!/usr/bin/env python3
"""Find the canonical library in an installed skill or repository template."""
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True

def main():
    skill = Path(__file__).resolve().parents[1]
    config = skill / 'repository.json'
    try:
        if config.exists() or config.is_symlink():
            root = Path(json.loads(config.read_text(encoding='utf-8'))['repository'])
            if not root.is_absolute():
                raise ValueError('repository path must be absolute')
        elif skill.parent.name == 'skill':
            root = skill.parent.parent
        else:
            raise ValueError('repository.json is missing')
        launcher = root / 'mwiki'
        if not launcher.is_file() or not (root / 'data/catalog.json').is_file():
            raise ValueError('repository is missing or incomplete: ' + str(root))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print('Metal KernelWiki: ' + str(exc) + '. Reinstall from the repository.', file=sys.stderr)
        return 1
    return subprocess.call([sys.executable, str(launcher), *sys.argv[1:]])

if __name__ == '__main__':
    raise SystemExit(main())
