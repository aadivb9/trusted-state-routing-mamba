"""Safety-of-inference and coverage tests for the new experiment harness."""
import importlib.util
from pathlib import Path
import sys,unittest
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(ROOT)]
from model import Encoder,inputs,lexical,symbolic
from run import operating_threshold


class Controls(unittest.TestCase):
 def test_inference_does_not_use_labels_ids_or_gold(self):
  r=dict(trusted='The study used 20 documents.',note='The study used 30 documents.',question='How many documents?',label=0,id='a',gold='20')
  altered=dict(r,label=1,id='b',gold='30',attack_family='secret',position='oracle offset')
  self.assertEqual(inputs(r),inputs(altered))
  np.testing.assert_array_equal(lexical([r]).toarray(),lexical([altered]).toarray())
  np.testing.assert_array_equal(symbolic([r]),symbolic([altered]))
  e=object.__new__(Encoder);e.cache={s:(np.ones(384),np.ones(384)*2) for s in inputs(r)}
  e.populate=lambda texts:None
  np.testing.assert_array_equal(e.transform([r]),e.transform([altered]))
  self.assertEqual(e.transform([r]).shape,(1,3072))

 def test_all_tokens_including_tail_reach_encoder(self):
  import torch
  captured=[]
  class Tokenizer:
   cls_token_id=101;sep_token_id=102
   def encode(self,text,**kwargs):return list(range(1000,1703))
   def build_inputs_with_special_tokens(self,chunk):
    captured.append(chunk);return [101]+chunk+[102]
   def pad(self,data,**kwargs):
    rows=data['input_ids'];captured.extend(r[1:-1] for r in rows);length=max(map(len,rows));ids=torch.tensor([r+[0]*(length-len(r)) for r in rows])
    return dict(input_ids=ids,attention_mask=(ids!=0).long())
  class Model:
   def __call__(self,**data):return SimpleNamespace(last_hidden_state=torch.ones((*data['input_ids'].shape,384)))
  e=object.__new__(Encoder);e.torch=torch;e.tokenizer=Tokenizer();e.model=Model();e.cache={}
  e.stats=dict(max_text_tokens=0,encoding_seconds=0.,unique_texts=0,windows=0)
  e.populate(['long input'])
  self.assertEqual(set(x for chunk in captured for x in chunk),set(range(1000,1703)))
  self.assertTrue(all(len(c)<=254 for c in captured))
  self.assertIn(1702,captured[-1]);self.assertEqual(e.stats['windows'],4)

 def test_threshold_enforces_false_positive_limit(self):
  y=np.array([0]*20+[1]*20);p=np.r_[np.linspace(.05,.95,20),np.linspace(.5,.99,20)]
  t,m=operating_threshold(y,p)
  self.assertLessEqual(((p>=t)&(y==0)).sum(),1)
  self.assertLessEqual(m['fpr'],.05)

 def test_threshold_can_reject_nothing(self):
  y=np.array([0,0,1,1]);p=np.array([.9,.9,.1,.1]);t,m=operating_threshold(y,p)
  self.assertGreater(t,p.max());self.assertEqual(m['fpr'],0.)


if __name__=='__main__':unittest.main()
