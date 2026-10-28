"""Fault injection into new empirical records must fail fresh recomputation."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from tools import verify_completion as gate

ROOT = Path(__file__).resolve().parents[1]


class CompletionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='completion-fault-')
        self.root = Path(self.temp.name)
        base = 'experiments/17_external_validation/'
        names = [base + name for name in ['protocol.json', 'runtime_amendment.json',
                 'runtime_amendment_v2.json', 'results.json', 'results_v2.json',
                 'prepared/development.json']]
        names += list(gate.read(ROOT, base + 'runtime_amendment_v2.json')['files'])
        names += ['archive/sealed_workspace/external_benchmark/audit_development.py']
        for name in names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, path)
        gate.external(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_relabeling_response_does_not_change_the_raw_score(self):
        path = self.root / 'experiments/17_external_validation/results_v2.json'
        data = json.loads(path.read_text())
        data['development'][0]['strict'] = data['development'][0]['gold']
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'saved score differs'):
            gate.external(self.root)

    def test_gate_total_cannot_be_changed_without_raw_evidence(self):
        path = self.root / 'experiments/17_external_validation/results_v2.json'
        data = json.loads(path.read_text())
        data['gate']['sst2']['strict_correct'] = 1
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'gate differs from raw'):
            gate.external(self.root)

    def test_missing_runtime_dependency_is_rejected(self):
        (self.root / 'src/runtime/serialization_v2.py').unlink()
        with self.assertRaises(FileNotFoundError):
            gate.external(self.root)

    def test_failed_timing_cannot_be_promoted_to_passing(self):
        base = 'experiments/18_runtime_validation/'
        names = [base + 'timing_protocol.json', base + 'timing_results.json']
        names += list(gate.read(ROOT, base + 'timing_protocol.json')['files'])
        for name in names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, path)
        audited = gate.timing(self.root)
        self.assertFalse(audited['full_protocol_completed'])
        path = self.root / base / 'timing_results.json'
        data = json.loads(path.read_text())
        data.update(status='complete', all_checks_passed=True, backbone_after=data['backbone_before'])
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'failed full timing protocol'):
            gate.timing(self.root)


if __name__ == '__main__':
    unittest.main()
