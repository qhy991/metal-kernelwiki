#!/usr/bin/env python3
"""CPU-only installation regression checks; all writes stay in temporary trees."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


class InstallSkillTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='metal-install-test-')
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.repo = self.base / 'checkout with spaces'
        shutil.copytree(ROOT, self.repo, ignore=shutil.ignore_patterns(
            '.git', '__pycache__', '*.pyc', 'knowledge', 'repository.json'))
        self.outside = self.base / 'unrelated cwd'
        self.outside.mkdir()
        self.skill = self.base / 'user skills' / 'metal-kernelwiki'
        self.installer = self.repo / 'scripts' / 'install_skill.py'
        result = self.run_python(self.installer, '--dest', self.skill)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.launcher = self.skill / 'scripts' / 'wiki.py'

    def run_python(self, script, *args, cwd=None):
        return subprocess.run(
            [sys.executable, '-B', str(script), *(str(arg) for arg in args)],
            cwd=cwd or self.outside, text=True, capture_output=True, timeout=20)

    def test_only_one_discoverable_skill(self):
        # Codex follows symlinks: a default, non-following walk missed the defect.
        discovered = []
        for directory, _, names in os.walk(self.skill, followlinks=True):
            if 'SKILL.md' in names:
                discovered.append((Path(directory) / 'SKILL.md').relative_to(self.skill))
        self.assertEqual(discovered, [Path('SKILL.md')])

    def test_retrieval_from_unrelated_cwd_and_live_content(self):
        for launcher in (self.launcher, self.repo / 'skill/metal-kernelwiki/scripts/wiki.py'):
            result = self.run_python(launcher, 'query', 'M4 GQA trace', '--limit', '1', '--json')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)[0]['id'], 'local-mlx-m4')
        canonical_page = self.repo / 'wiki' / 'measurement.md'
        self.assertTrue((self.skill / 'knowledge/wiki/measurement.md').samefile(canonical_page))
        marker = '\nTemporary canonical-content change for installation regression.\n'
        with canonical_page.open('a', encoding='utf-8') as handle:
            handle.write(marker)
        result = self.run_python(self.launcher, 'get', 'measurement')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(marker, result.stdout)
        result = self.run_python(self.launcher, 'validate')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_existing_install_is_not_overwritten(self):
        config = self.skill / 'repository.json'
        before = config.read_bytes()
        result = self.run_python(self.installer, '--dest', self.skill)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('refusing to overwrite', result.stderr)
        self.assertEqual(config.read_bytes(), before)

    def test_missing_or_invalid_locator_does_not_use_cwd(self):
        config = self.skill / 'repository.json'
        for value in (None, '{broken JSON', '{"repository": "relative/path"}', '{}'):
            if value is None:
                config.unlink()
            else:
                config.write_text(value, encoding='utf-8')
            for cwd in (self.repo, self.outside):
                result = self.run_python(self.launcher, 'get', 'measurement', cwd=cwd)
                self.assertEqual(result.returncode, 1)
                self.assertIn('Reinstall from the repository', result.stderr)
                self.assertNotIn('Traceback', result.stderr)


if __name__ == '__main__':
    unittest.main()
