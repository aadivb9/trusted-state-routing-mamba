import unittest

from evaluate_generalization_boundary_fix import token_ranges


class TokenBoundaryTests(unittest.TestCase):
    def test_whitespace_merge_is_assigned_to_following_span(self):
        self.assertEqual(token_ranges(['Fact ', '\n\nNote'], [(0, 4), (4, 7), (7, 11)]),
                         [(0, 1), (1, 3)])

    def test_empty_context_produces_empty_first_chunk(self):
        self.assertEqual(token_ranges(['', '\nQuestion'], [(0, 1), (1, 9)]), [(0, 0), (0, 2)])

    def test_content_cannot_move_across_trust_boundary(self):
        with self.assertRaisesRegex(ValueError, 'non-whitespace'):
            token_ranges(['Fact', 'Note'], [(0, 8)])

    def test_aligned_tokens_keep_their_original_spans(self):
        self.assertEqual(token_ranges(['Fact', '\nNote'], [(0, 4), (4, 5), (5, 9)]),
                         [(0, 1), (1, 3)])


if __name__ == '__main__':
    unittest.main()
