"""Fault injection for saved follow-up evidence; no model download or inference."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('followup_summary',ROOT/'tools/followup_summary.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class FollowupFaultTests(unittest.TestCase):
    def detector_copy(self,dest):
        folder='experiments/19_embedding_followup/'
        protocol=json.loads((ROOT/(folder+'protocol.json')).read_text())
        names=set(protocol['files'])|{folder+n for n in ['protocol.json','results.json','selection.json','frozen_heads.json','calibration_replay.json','replay_calibration.py','preparation_audit.json','derived_official_test.jsonl']}
        provenance=json.loads((ROOT/(folder+'raw/provenance.json')).read_text())
        names.update(folder+'raw/'+name for name in provenance['files'])
        for name in names:
            path=dest/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,path)
        return dest/(folder+'results.json')

    def test_edited_detection_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp);path=self.detector_copy(dest)
            result=json.loads(path.read_text());result['final']['minilm/document']['predictions'].pop();path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,'score row'):
                audit.detector(dest)

    def test_changed_rejection_mask_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp);path=self.detector_copy(dest)
            result=json.loads(path.read_text());result['final']['minilm/document']['predictions'][0]['rejected_spans'][0]=True;path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,'frozen threshold'):
                audit.detector(dest)

    def test_missing_frozen_dependency_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp);path=self.detector_copy(dest)
            (dest/'src/routing/embedding_followup.py').unlink()
            with self.assertRaises(FileNotFoundError):
                audit.detector(dest)

    def test_attack_validity_is_unchanged_by_already_completed_defended_rows(self):
        recovered=json.loads((ROOT/'experiments/20_sst2_followup/progress_snapshots/generations_534.json').read_text())
        original=[r for r in recovered['rows'] if r['role']!='attack' or r['arm']=='undefended']
        self.assertEqual(audit.attack_gate(original),recovered['attack_validity'])
        self.assertEqual(audit.attack_gate(recovered['rows']),recovered['attack_validity'])

    def test_auc_ties_and_reversed_ranking(self):
        self.assertEqual(audit.auc([0,1],[.5,.5]),.5)
        self.assertEqual(audit.auc([0,1],[.9,.1]),0)
        self.assertEqual(audit.auc([0,1],[.1,.9]),1)


if __name__=='__main__':unittest.main()
