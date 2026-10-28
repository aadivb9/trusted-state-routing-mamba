"""Verify test-set controls, strict answer parsing, and paired statistics."""
from collections import Counter
import unittest

import generate_text_router_benchmark as data
import evaluate_text_router_benchmark as evaluation


class FreshControls(unittest.TestCase):
    def test_answer_positions_and_test_families_are_balanced(self):
        cases = data.source_cases()
        self.assertEqual(len(cases),100)
        self.assertEqual(len({case['study'] for case in cases}),100)
        self.assertEqual(Counter(case['spec']['values'].index(case['actual']) for case in cases),{0:50,1:50})
        self.assertEqual(set(Counter(case['attack_style'] for case in cases).values()),{20})
        self.assertEqual(set(Counter(case['safe_style'] for case in cases).values()),{20})

    def test_useful_safe_cases_remove_only_the_queried_fact(self):
        cases = [case for case in data.source_cases() if case['requires_safe_span']]
        self.assertEqual(len(cases),20)
        for case in cases:
            self.assertIn(case['target_fact'],case['full_context'])
            self.assertNotIn(case['target_fact'],case['safe_context'])
            self.assertEqual(len(case['full_context'].split('. '))-len(case['safe_context'].split('. ')),1)

    def test_parser_does_not_accept_a_value_buried_in_an_explanation(self):
        self.assertEqual(evaluation.parse_free(' 126 documents.','documents'),'126')
        self.assertEqual(evaluation.parse_free(' GPU.','hardware'),'GPU')
        self.assertEqual(evaluation.parse_free(' B was fastest.','method'),'B')
        self.assertIsNone(evaluation.parse_free('I would say 126.','documents'))
        self.assertIsNone(evaluation.parse_free('CPUs are useful.','hardware'))
        self.assertIsNone(evaluation.parse_free('Because the study says so.','method'))

    def test_paired_statistics_keep_source_matching(self):
        first = {'source_one':False,'source_two':True}
        second = {'source_two':True,'source_one':True}
        result = evaluation.paired_difference(first,second,samples=1000)
        self.assertEqual(result['difference'],0.5)
        unchanged = evaluation.paired_difference(first,first,samples=100)
        self.assertEqual(unchanged['bootstrap_95'],[0.,0.])
        with self.assertRaises(AssertionError):
            evaluation.paired_difference(first,{'other':True})


if __name__=='__main__':
    unittest.main()
