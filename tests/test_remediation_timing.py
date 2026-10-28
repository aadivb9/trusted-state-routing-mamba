"""Paired timing summaries must retain failures and exact repetition coverage."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    'timing_driver', Path(__file__).resolve().parents[1] /
    'experiments/18_runtime_validation/run_timing.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TimingSummaryTests(unittest.TestCase):
    def setUp(self):
        self.plan = {'prefix_tokens': [128], 'timed_repetitions': 2,
                     'modes': ['original', 'forced_accept', 'forced_reject',
                               'identical_decision_filtering'],
                     'bootstrap_resamples': 100, 'paired_order_seed': 20261022}
        self.rows = [{'prefix_tokens': 128, 'repetition': repetition,
                      'mode': mode, 'elapsed_seconds': factor * (repetition + 1),
                      'generated_tokens': [4, 5], 'peak_allocated_bytes': 100,
                      'peak_reserved_bytes': 200}
                     for repetition in range(2)
                     for mode, factor in [('original', 1), ('forced_accept', 2),
                                          ('forced_reject', 6), ('identical_decision_filtering', 3)]]

    def test_shuffled_rows_preserve_paired_ratios_and_percentile(self):
        summaries, pairs = MODULE.summarize(list(reversed(self.rows)), self.plan)
        self.assertEqual(summaries[0]['median_seconds'], 1.5)
        self.assertEqual(summaries[0]['p95_seconds_nearest_rank'], 2)
        for pair in pairs:
            self.assertEqual(pair['median_paired_time_ratio'], 2)
            self.assertEqual(pair['bootstrap_percentile_95_interval'], [2, 2])
            self.assertTrue(pair['all_generated_tokens_match'])

    def test_token_mismatch_cannot_be_reported_as_equivalence(self):
        self.rows[1]['generated_tokens'] = [9, 9]
        _, pairs = MODULE.summarize(self.rows, self.plan)
        self.assertFalse(pairs[0]['all_generated_tokens_match'])
        self.assertEqual(pairs[0]['matched_pairs'], 1)

    def test_duplicate_or_missing_pair_is_rejected(self):
        self.rows[0] = dict(self.rows[4])
        with self.assertRaisesRegex(ValueError, 'Incomplete or duplicate'):
            MODULE.summarize(self.rows, self.plan)
        with self.assertRaisesRegex(ValueError, 'row count'):
            MODULE.summarize(self.rows[:-1], self.plan)


if __name__ == '__main__':
    unittest.main()
