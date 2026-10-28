"""A public shuffle seed must not reproduce the exported opaque identifiers."""
import json,unittest
from pathlib import Path
from src.validation.blinded_review_v2 import make_packet

ROOT=Path(__file__).resolve().parents[1]


class BlindingV2Tests(unittest.TestCase):
    def records(self):return json.loads((ROOT/'archive/sealed_workspace/generalization_experiment/dataset.json').read_text())['records']

    def test_ids_are_independent_of_shuffle_seed(self):
        a,_=make_packet(self.records()[:2],20261022);b,_=make_packet(self.records()[:2],20261022)
        ids=lambda p:{n['note_id'] for i in p['items'] for n in i['notes']}
        self.assertFalse(ids(a)&ids(b))

    def test_export_has_private_seed_and_balanced_candidate_positions(self):
        folder=ROOT/'experiments/16_blinded_annotation'
        status=json.loads((folder/'status.json').read_text());private=json.loads((folder/'private/mapping.json').read_text());packet=json.loads((folder/'reviewer_packet.json').read_text())
        self.assertEqual(status['packet_version'],2)
        self.assertGreater(private['seed'].bit_length(),128)
        self.assertNotIn(str(private['seed']),json.dumps(packet))
        self.assertNotIn('seed',packet)
        positions=[next(j for j,c in enumerate(i['candidates']) if c['candidate_id']==private['items'][i['item_id']]['gold_candidate_id']) for i in packet['items']]
        self.assertEqual(positions.count(0),16)
        self.assertEqual(positions.count(1),16)


if __name__=='__main__':unittest.main()
