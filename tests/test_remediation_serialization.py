"""Verify token filtering preserves the exact selected stream for cache controls."""
import unittest
from src.runtime.serialization import chunks,selected_tokens


class SerializationTests(unittest.TestCase):
    def test_complete_prompt_is_tokenized_once(self):
        calls=[]
        def tokenizer(text,**kwargs):
            calls.append(text)
            return {'input_ids':list(range(len(text))),'offset_mapping':[(i,i+1) for i in range(len(text))]}
        prompt,parts=chunks('Instruction','Note',tokenizer)
        self.assertEqual(calls,[prompt])
        self.assertEqual(selected_tokens(parts,True),list(range(len(prompt))))
        self.assertEqual(selected_tokens(parts,False),parts[0]+parts[2])

    def test_unsafe_non_whitespace_boundary_is_rejected(self):
        def tokenizer(text,**kwargs):return {'input_ids':[9],'offset_mapping':[(0,len(text))]}
        with self.assertRaisesRegex(ValueError,'trust boundary'):chunks('Instruction','Note',tokenizer)

    def test_empty_note_does_not_invent_an_update(self):
        def tokenizer(text,**kwargs):return {'input_ids':list(range(len(text))),'offset_mapping':[(i,i+1) for i in range(len(text))]}
        _,parts=chunks('Instruction','',tokenizer)
        self.assertEqual(parts[1],[])
        self.assertEqual(selected_tokens(parts,True),selected_tokens(parts,False))


if __name__=='__main__':unittest.main()
