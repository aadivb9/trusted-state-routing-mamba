"""Fault-injection checks in temporary copies; no production evidence is edited."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile
import check


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='forge-test-')
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def copy(self, name):
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(check.ROOT / name, target)
        return target

    def test_changed_count_is_rejected(self):
        for e in check.EXPERIMENTS:
            self.copy(f'{check.REV}/{e}/results.json')
        self.copy('research_closeout/run_03/results.json')
        path = self.copy('paper/manuscript.md')
        check.paper_tables(self.root)
        path.write_text(path.read_text().replace('| clean | 16/16 | 7/16 |', '| clean | 15/16 | 7/16 |'))
        with self.assertRaisesRegex(ValueError, 'HiSPA table mismatch'):
            check.paper_tables(self.root)

    def test_stale_pdf_is_rejected(self):
        record = json.loads(self.copy('paper/build_verification.json').read_text())
        for name in {**record['inputs'], **record['outputs']}:
            self.copy(name)
        check.render_freshness(self.root)
        with (self.root / 'paper/manuscript.pdf').open('ab') as stream:
            stream.write(b'\nstale edit')
        with self.assertRaisesRegex(ValueError, 'Stale render/source'):
            check.render_freshness(self.root)

    def test_missing_archive_dependency_is_rejected(self):
        archive = self.root / 'revision_experiments_evidence.zip'
        with zipfile.ZipFile(check.ROOT / archive.name) as source, zipfile.ZipFile(archive, 'w') as target:
            for name in source.namelist():
                if name != 'final_hispa_audit/results.json':
                    target.writestr(name, source.read(name))
        with self.assertRaisesRegex(ValueError, 'final_hispa_audit/results.json'):
            check.isolated_archive_audits(self.root)

    def test_stale_status_is_rejected(self):
        for e in check.EXPERIMENTS:
            self.copy(f'{check.REV}/{e}/results.json')
        for name in check.ENTRY_POINTS:
            self.copy(name)
        check.statuses(self.root)
        path = self.root / 'README.md'
        path.write_text(path.read_text().replace('112 HiSPA', '111 HiSPA'))
        with self.assertRaisesRegex(ValueError, 'Stale/missing current status'):
            check.statuses(self.root)

    def test_unaudited_narrative_edit_is_rejected(self):
        self.copy('.forge/narrative_review.json')
        for name in check.REVIEWED:
            self.copy(name)
        check.review_freshness(self.root)
        with (self.root / 'revision_experiments/REPORT.md').open('a') as stream:
            stream.write('\nNew unreviewed conclusion.\n')
        with self.assertRaisesRegex(ValueError, 'Narrative changed since semantic review'):
            check.review_freshness(self.root)


if __name__ == '__main__':
    unittest.main()
