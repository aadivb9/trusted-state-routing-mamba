"""Control tests for the new development/holdout and observable-text boundaries."""
import copy,importlib.util,json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'experiments/20_sst2_followup'),str(ROOT/'archive/sealed_workspace')]
from src.routing.embedding_followup import segment,Encoder
from prompts import FORMATS,render_parts
spec=importlib.util.spec_from_file_location('followup_screen',ROOT/'experiments/20_sst2_followup/run_screen.py');screen=importlib.util.module_from_spec(spec);spec.loader.exec_module(screen)

class ObservableBoundaryTests(unittest.TestCase):
 def test_segmentation_preserves_all_bytes_and_word_limits(self):
  for text in ['A sentence. Another sentence!\n\nFinal line.',' émoji 🦉 '+('word '*200),'\n\n','']:
   spans=segment(text);self.assertEqual(''.join(x['text'] for x in spans),text)
   self.assertTrue(all(len(s['text'].split())<=64 for s in spans))
 def test_model_features_ignore_labels_offsets_and_identifiers(self):
  e=Encoder.__new__(Encoder);e.cache={s:(np.array([i+.1,i+.2]),np.array([i+.3,i+.4])) for i,s in enumerate(['trusted','question','one','two'])}
  a={'trusted':'trusted','question':'question','spans':[{'text':'one','label':0},{'text':'two','label':1}],'label':1,'attack_interval':[0,2],'source_id':'first'}
  b=copy.deepcopy(a);b.update(label=0,attack_interval=[500,900],source_id='different');b['spans'][0]['label']=1;b['spans'][1]['label']=0
  for granularity in ['document','span']:np.testing.assert_array_equal(e.features(a,granularity),e.features(b,granularity))

class PromptGateTests(unittest.TestCase):
 def test_only_explicit_training_demonstration_labels_enter_prompts(self):
  data={'instruction':'Classify sentiment.','demonstrations':[{'text':'A good film.','label':1},{'text':'A poor film.','label':0}],'current_gold':'LEAK_MARKER'}
  for fmt in FORMATS:
   parts=render_parts(data,fmt,'Current review.');self.assertNotIn('LEAK_MARKER',''.join(parts));self.assertEqual(parts[1],'Current review.')
 def test_unknown_formats_are_rejected(self):
  with self.assertRaises(ValueError):render_parts({'instruction':'Classify.'},'new_after_outcomes','text')
 def test_validity_gate_is_not_replaced_by_accuracy(self):
  rows=[]
  for i in range(40):
   gold=i%2;strict=gold if i<30 else (1-gold if i<37 else None);rows.append({'gold':gold,'strict':strict,'official':strict})
  self.assertTrue(screen.score(rows,0)['passed']);rows[36]['strict']=None
  self.assertFalse(screen.score(rows,0)['passed'])
 def test_new_roles_have_no_exact_source_overlap(self):
  data=json.loads((ROOT/'experiments/20_sst2_followup/prepared.json').read_text());sets=[{r['source_id'] for r in data[name]} for name in ['demonstrations','development','screen','targets','injections']]
  for i,a in enumerate(sets):
   for b in sets[:i]:self.assertFalse(a&b)

if __name__=='__main__':unittest.main()
