"""Post-run frozen-router reconstruction; never refits or changes a decision."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.routing.embedding_followup import Encoder
spec=importlib.util.spec_from_file_location('frozen_predict',ROOT/'experiments/19_embedding_followup/run.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def main():
    path=OUT/'mask_replay.json';assert not path.exists(),'Preserve prior replay'
    masks=json.loads((OUT/'masks.json').read_text());data=json.loads((OUT/'prepared.json').read_text());head=json.loads((ROOT/'experiments/19_embedding_followup/frozen_heads.json').read_text())[masks['head']]
    records=[dict(r,trusted=data['instruction'],question=data['instruction']) for r in masks['records']]
    encoder=Encoder(head['encoder'],'mps');encoder.populate_records(records)
    max_difference=0.;mismatches=[]
    for r in records:
        values=m.predict(encoder,head,r);max_difference=max(max_difference,max(abs(a-b) for a,b in zip(values,r['scores'])))
        if [v>=head['threshold'] for v in values]!=r['learned_rejected']:mismatches.append(r['id'])
    result=dict(kind='Post-run same-encoder frozen-head mask reconstruction; not independent replication',status='passed' if not mismatches and max_difference<=1e-7 else 'mismatch',records=len(records),max_probability_difference=max_difference,decision_mismatches=mismatches,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),masks_sha256=hashlib.sha256((OUT/'masks.json').read_bytes()).hexdigest())
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
