"""Frozen calibration-selected router; no target labels/attack offsets in decisions."""
import hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from src.routing.embedding_followup import Encoder,segment
from src.runtime.region_serialization import serialize,retained
import importlib.util
spec=importlib.util.spec_from_file_location('fixed_embedding_runner',ROOT/'experiments/19_embedding_followup/run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
RULE=re.compile(r'\b(?:ignore|disregard|forget)\b.{0,100}\b(?:previous|instruction|task)\b|\b(?:system|assistant|user)\s*:|\b(?:instead|override)\b|\b(?:output|return|answer)\s+(?:only|exactly)\b',re.I|re.S)
def main():
 assert not (OUT/'masks.json').exists(),'Preserve frozen masks'
 data=json.loads((OUT/'prepared.json').read_text());folder=ROOT/'experiments/19_embedding_followup';selection=json.loads((folder/'selection.json').read_text());heads=json.loads((folder/'frozen_heads.json').read_text());head=heads[selection['head']]
 assert head['granularity']=='span','Predefined final architecture requires the calibration-selected span policy; do not substitute using final results'
 records=[]
 for role,rows in [('clean',data['targets']),('attack',data['attacks']),('injected_baseline',data['injections'])]:
  for row in rows:
   records.append(dict(id=role+'/'+row.get('id',row.get('source_id')),role=role,source_id=row.get('target_source_id',row.get('source_id')),gold=row.get('gold',row.get('label')),injected_gold=row.get('injected_gold'),family=row.get('family'),text=row['text'],trusted=data['instruction'],question=data['instruction'],spans=segment(row['text'])))
 e=Encoder(head['encoder'],'mps');e.populate_records(records)
 for r in records:
  r['scores']=m.predict(e,head,r);r['learned_rejected']=[s>=head['threshold'] for s in r['scores']];r['rule_rejected']=[bool(RULE.search(s['text'])) for s in r['spans']]
  del r['trusted'];del r['question']
 output=dict(records=records,head=selection['head'],threshold=head['threshold'],confidence_qualified=False,selection_sha256=hashlib.sha256((folder/'selection.json').read_bytes()).hexdigest(),heads_sha256=selection['heads_sha256'],prepared_sha256=hashlib.sha256((OUT/'prepared.json').read_bytes()).hexdigest(),scope='Masks depend only on trusted task instruction, external span text and fixed head. Source IDs, labels, families and attack offsets are not inputs. No original clean counterpart used.')
 (OUT/'masks.json').write_text(json.dumps(output,indent=2)+'\n');print('Frozen masks',len(records))
if __name__=='__main__':main()
