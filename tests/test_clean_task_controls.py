"""CPU checks for experimental controls; GPU scoring checks run in the diagnostic."""
import unittest
from collections import Counter

import clean_baseline_diagnostic as diagnostic


class ExperimentalControls(unittest.TestCase):
    def test_counterfactual_pairs_hold_question_and_choices_fixed(self):
        records = diagnostic.confirmation_records()
        self.assertEqual(len(records), 128)
        for first, second in zip(records[::2], records[1::2]):
            self.assertEqual(first['question'], second['question'])
            self.assertEqual(first['pair_id'], second['pair_id'])
            # Four source sentences; exactly one supporting sentence changes.
            changed = sum(a != b for a, b in zip(
                first['trusted_context'].split('. '), second['trusted_context'].split('. ')))
            self.assertEqual(changed, 1)
            for mode in diagnostic.FORMATS:
                prompt_a, choices_a, gold_a = diagnostic.render(first, mode)
                prompt_b, choices_b, gold_b = diagnostic.render(second, mode)
                self.assertNotEqual(prompt_a, prompt_b)
                self.assertEqual(choices_a, choices_b)
                self.assertNotEqual(gold_a, gold_b)

    def test_no_context_control_is_indistinguishable_with_opposite_labels(self):
        records = diagnostic.confirmation_records()
        for first, second in zip(records[::2], records[1::2]):
            for mode in diagnostic.FORMATS:
                a = diagnostic.render(first, mode, empty_context=True)
                b = diagnostic.render(second, mode, empty_context=True)
                self.assertEqual(a[:2], b[:2])
                self.assertNotEqual(a[2], b[2])

    def test_balanced_confirmation_targets_for_every_format_and_fact_type(self):
        records = diagnostic.confirmation_records()
        for mode in diagnostic.FORMATS:
            for slot in ['documents', 'hardware', 'method', 'runs']:
                gold = Counter(diagnostic.render(r, mode)[2] for r in records if r['slot'] == slot)
                self.assertEqual(gold, {0: 16, 1: 16})


if __name__ == '__main__':
    unittest.main()
