"""CPU checks for experimental controls, routing inputs, and sealed provenance."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import build_generalization_data as data
import evaluate_generalization as evaluation
import write_generalization_report as report

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'generalization_experiment'


class GeneralizationControls(unittest.TestCase):
    def test_source_tasks_pass_structural_controls(self):
        papers = json.loads((OUT / 'source_abstracts.json').read_text())
        tasks = json.loads((OUT / 'accepted_tasks.json').read_text())
        protocol = json.loads((OUT / 'protocol.json').read_text())
        self.assertEqual([p['id'] for p in papers], protocol['paper_ids'])
        self.assertFalse(set(protocol['paper_ids']) & set(protocol['excluded_previously_evaluated_papers']))
        self.assertEqual(len(tasks), 32)
        for paper in papers:
            self.assertEqual(data.validate(tasks[str(paper['id'])], paper), [])

    def test_layout_preserves_attack_order_and_removed_evidence(self):
        record = {'abstract': 'Full evidence.', 'context_without_support': 'Remaining text.'}
        self.assertEqual(evaluation.layout(record, 'empty_context'), ('', []))
        self.assertEqual(evaluation.layout(record, 'safe_addition')[0], 'Remaining text.')
        self.assertEqual(evaluation.layout(record, 'mixed_note')[0], 'Remaining text.')
        self.assertEqual(evaluation.layout(record, 'attack_early')[1][0], 'malicious_note')
        self.assertEqual(evaluation.layout(record, 'attack_late')[1][-1], 'malicious_note')

    def test_router_receives_no_labels_choices_or_prior_notes(self):
        record = {'abstract': 'Source', 'stem': 'Question', 'choices': ['secret', 'foil'],
                  'answer': 'secret', 'gold_index': 0, 'malicious_note': 'Attack',
                  'support_note': 'Support', 'benign_note': 'Benign'}
        with patch.object(evaluation.routing, 'route', return_value={'rejected': False}) as route:
            evaluation.decisions({}, record, 'attack_early')
        self.assertEqual([call.args for call in route.call_args_list],
                         [({}, 'Source', note, 'Question') for note in ['Attack', 'Support', 'Benign']])

    def test_strict_free_parser_rejects_explanation_and_word_prefix(self):
        self.assertEqual(evaluation.free_prediction('  Graph   networks. Explanation', ['graph networks', 'trees']), 0)
        self.assertIsNone(evaluation.free_prediction('The answer is trees.', ['graph networks', 'trees']))
        self.assertIsNone(evaluation.free_prediction('treesX', ['graph networks', 'trees']))

    def test_paired_interval_requires_same_papers_and_detects_no_gain(self):
        with self.assertRaises(AssertionError):
            evaluation.paired({1: True}, {2: True})
        self.assertEqual(evaluation.paired({1: True, 2: False}, {1: True, 2: False})['bootstrap_95'], [0, 0])
        self.assertEqual(evaluation.paired({1: False, 2: False}, {1: True, 2: True})['bootstrap_95'], [1, 1])

    def test_seal_prevents_input_mutation(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(data, 'OUT', Path(directory)):
            (Path(directory) / 'seal.json').write_text('{}')
            for command in [data.generate, data.audit, data.assemble]:
                with self.assertRaisesRegex(RuntimeError, 'sealed'):
                    command()

    def test_later_manifest_cannot_override_original_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / 'generalization_experiment'
            out.mkdir()
            (root / 'frozen.txt').write_text('tampered')
            digest = hashlib.sha256(b'tampered').hexdigest()
            (out / 'protocol.json').write_text(json.dumps({'frozen_files': {'frozen.txt': 'original'}}))
            (out / 'seal.json').write_text(json.dumps({'files': {'frozen.txt': digest}}))
            with patch.object(evaluation, 'ROOT', root), patch.object(evaluation, 'OUT', out):
                with self.assertRaises(AssertionError):
                    evaluation.verify()

    def test_report_handles_stopped_clean_gate_without_defense_results(self):
        # Synthetic fixture lives only in a temporary directory, never in run artifacts.
        metric = {'correct': 1, 'n': 2, 'accuracy': .5}
        result = {'status': 'stopped_clean_gate', 'clean_gate': {
            'clean': metric, 'empty_context': metric, 'passed': False,
            'context_gain': {'difference': 0, 'bootstrap_95': [0, 0]}},
            'manifest': {'backbone_before_sha256': 'fixture', 'gpu': 'fixture'},
            'backbone_after_sha256': 'fixture'}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            for name, value in [('dataset.json', {'records': [{}, {}]}),
                                ('semantic_audit.json', {'n': 2, 'unresolved': []}),
                                ('results.json', result)]:
                (out / name).write_text(json.dumps(value))
            with patch.object(report, 'OUT', out):
                report.render()
            text = (out / 'REPORT.md').read_text()
            self.assertIn('stopped_clean_gate', text)
            self.assertIn('No robustness gain is established', text)
            self.assertNotIn('## Answer accuracy', text)


if __name__ == '__main__':
    unittest.main()
