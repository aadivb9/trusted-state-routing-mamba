"""Fault injection: fresh hash stamps cannot hide false paper/completion claims."""
import importlib.util,json,shutil,tempfile,unittest,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('remediation_gate',ROOT/'tools/verify_remediation.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)


class RemediationGateRegressions(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='remediation-fault-');self.root=Path(self.tmp.name)

    def tearDown(self):self.tmp.cleanup()

    def copy(self,name):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,p);return p

    def paper_fixture(self):
        b=gate.read(ROOT,'paper/build_verification.json');review=gate.read(ROOT,'remediation/narrative_review.json')
        names=set(b['inputs'])|set(b['outputs'])|set(review['files'])|{'paper/build_verification.json','remediation/narrative_review.json','remediation/finding_status.json'}
        for name in names:self.copy(name)
        gate.active_paper(self.root)

    def test_false_count_rejected_even_with_refreshed_build_stamp(self):
        self.paper_fixture();p=self.root/'paper/manuscript.md'
        p.write_text(p.read_text().replace('| Trained cascade | 99/100','| Trained cascade | 98/100'))
        b=gate.read(self.root,'paper/build_verification.json');b['inputs']['paper/manuscript.md']=gate.sha(p)
        (self.root/'paper/build_verification.json').write_text(json.dumps(b))
        with self.assertRaisesRegex(ValueError,'Active router table mismatch'):gate.active_paper(self.root)

    def test_stale_pdf_rejected(self):
        self.paper_fixture()
        with (self.root/'paper/manuscript.pdf').open('ab') as f:f.write(b'fake update')
        with self.assertRaisesRegex(ValueError,'Active paper stale'):gate.active_paper(self.root)

    def test_false_efficacy_completion_rejected_even_after_narrative_stamp(self):
        self.paper_fixture();p=self.root/'remediation/finding_status.json'
        status=json.loads(p.read_text());status['external_security_efficacy_established']=True;p.write_text(json.dumps(status))
        review=gate.read(self.root,'remediation/narrative_review.json');review['files']['remediation/finding_status.json']=gate.sha(p)
        (self.root/'remediation/narrative_review.json').write_text(json.dumps(review))
        with self.assertRaisesRegex(ValueError,'Unsupported completion claim'):gate.active_paper(self.root)

    def test_private_mapping_cannot_enter_reviewer_zip(self):
        base='experiments/16_blinded_annotation/'
        for name in ['status.json','private/mapping.json','reviewer_packet.json','reviewer_packet.zip']:self.copy(base+name)
        amendment=json.loads(self.copy(base+'blinding_amendment.json').read_text())
        for name in amendment['files']:self.copy(base+name)
        status=gate.read(self.root,base+'status.json')
        self.copy(status['review_artifact'])
        gate.blinding(self.root)
        with zipfile.ZipFile(self.root/base/'reviewer_packet.zip','a') as z:z.writestr('private/mapping.json','{}')
        with self.assertRaisesRegex(ValueError,'leaks private'):gate.blinding(self.root)


if __name__=='__main__':unittest.main()
