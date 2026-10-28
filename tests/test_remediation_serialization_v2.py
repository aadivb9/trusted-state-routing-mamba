"""Boundary whitespace is owned without trusting any untrusted content."""
import unittest
from src.runtime.serialization_v2 import chunks, selected_tokens


class BoundaryOwnershipTests(unittest.TestCase):
    def test_leading_space_and_untrusted_word_belong_to_note(self):
        instruction, note = 'Classify.', 'Hello'
        prefix = instruction + '\nText: '
        boundary = len(prefix)
        def tokenizer(prompt, **kwargs):
            return {'input_ids': [1, 2, 3],
                    'offset_mapping': [(0, boundary - 1),
                                       (boundary - 1, boundary + len(note)),
                                       (boundary + len(note), len(prompt))]}
        prompt, parts = chunks(instruction, note, tokenizer)
        self.assertEqual(prompt, prefix + note + '\n')
        self.assertEqual(parts, [[1], [2], [3]])
        self.assertEqual(selected_tokens(parts, True), [1, 2, 3])
        self.assertEqual(selected_tokens(parts, False), [1, 3])

    def test_note_with_suffix_whitespace_stays_untrusted(self):
        def tokenizer(prompt, **kwargs):
            boundary = prompt.index('Hello')
            return {'input_ids': [1, 2],
                    'offset_mapping': [(0, boundary), (boundary, len(prompt))]}
        _, parts = chunks('Classify.', 'Hello', tokenizer)
        self.assertEqual(parts, [[1], [2], []])

    def test_real_mixed_content_boundary_still_rejected(self):
        def tokenizer(prompt, **kwargs):
            return {'input_ids': [1], 'offset_mapping': [(0, len(prompt))]}
        with self.assertRaisesRegex(ValueError, 'trust boundary'):
            chunks('Classify.', 'Hello', tokenizer)

    def test_empty_note_does_not_create_content(self):
        def tokenizer(prompt, **kwargs):
            return {'input_ids': list(range(len(prompt))),
                    'offset_mapping': [(i, i + 1) for i in range(len(prompt))]}
        _, parts = chunks('Classify.', '', tokenizer)
        self.assertEqual(parts[1], [])
        self.assertEqual(selected_tokens(parts, True), selected_tokens(parts, False))


if __name__ == '__main__':
    unittest.main()
