"""Frozen-head calibration-score reconstruction; no refitting or threshold changes."""
import gc,hashlib,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from src.routing.embedding_followup import Encoder,MODELS
spec=importlib.util.spec_from_file_location('sealed_followup_runner',OUT/'run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def main():
 import torch
 assert not (OUT/'calibration_replay.json').exists(),'Preserve replay evidence'
 result=json.loads((OUT/'results.json').read_text());heads=json.loads((OUT/'frozen_heads.json').read_text());data=json.loads((OUT/'prepared.json').read_text());training=[r for r in data if r['split']=='train'];cal=[r for r in data if r['split']=='calibration']
 report={'kind':'Post-run frozen-head score reconstruction, not original raw-score logging','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'heads_sha256':hashlib.sha256((OUT/'frozen_heads.json').read_bytes()).hexdigest(),'prepared_sha256':hashlib.sha256((OUT/'prepared.json').read_bytes()).hexdigest(),'heads':{}}
 for name in MODELS:
  e=Encoder(name,'mps');e.populate_records(training+cal)
  for granularity in ['document','span']:
   key=name+'/'+granularity;head=heads[key];risks=[m.predict(e,head,r) for r in cal];rows=m.aggregate(cal,risks,granularity);metrics=m.threshold_metrics(rows,head['threshold']);original=result['heads'][key]['point_operating']['domains']
   parity=all(all(metrics[d][k]==v for k,v in original[d].items() if k!='fpr_upper') for d in metrics)
   report['heads'][key]={'original_decision_metrics_match':parity,'metrics':metrics,'scores':[dict(id=r['id'],domain=r['domain'],source_group=r['source_group'],label=r['label'],span_scores=s) for r,s in zip(cal,risks)]}
   (OUT/'calibration_replay.json').write_text(json.dumps(report,indent=2)+'\n');print('REPLAY',key,parity,flush=True)
  del e;gc.collect();torch.mps.empty_cache()
 report['status']='passed' if all(h['original_decision_metrics_match'] for h in report['heads'].values()) else 'decision_metrics_differ'
 (OUT/'calibration_replay.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
