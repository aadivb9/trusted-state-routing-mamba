"""Bounded prompt development followed by a fresh fixed competence screen."""
import hashlib,json,math,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'archive/sealed_workspace'),str(OUT)]
from prompts import FORMATS,render_parts
from src.runtime.cuda_models import load
from external_benchmark.audit_development import predictions

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,data):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)
def score(rows,empty):
 n=len(rows);correct=sum(r['strict']==r['gold'] for r in rows);valid=sum(r['strict'] is not None for r in rows)
 gain=(correct-sum(empty==r['gold'] for r in rows))/n
 return dict(n=n,strict_correct=correct,strict_valid=valid,input_gain=gain,published_correct=sum(r['official']==r['gold'] for r in rows),passed=correct>=math.ceil(.75*n) and valid>=math.ceil(22/24*n) and gain>=.2)

def main():
 import torch,importlib.metadata
 p=json.loads((OUT/'protocol.json').read_text());data=json.loads((OUT/'prepared.json').read_text())
 for name,h in p['files'].items():assert sha(ROOT/name)==h,'Frozen input changed: '+name
 assert not (OUT/'screen_results.json').exists(),'Preserve prior screen'
 result=dict(status='running',protocol_sha256=sha(OUT/'protocol.json'),development=[],instruction_only={},confirmation=[],selection=None,versions={n:importlib.metadata.version(n) for n in ['torch','transformers','accelerate','huggingface-hub']})
 path=OUT/'screen_results.json';save(path,result)
 try:
  assert torch.cuda.is_available() and torch.__version__.startswith('2.11.0')
  for name,version in p['runtime_versions'].items():assert importlib.metadata.version(name)==version,(name,version)
  import dual_state_mamba as core
  model,tok=load('mamba_2_8b');result.update(gpu=torch.cuda.get_device_name(0),backbone_before=core.model_fingerprint(model));save(path,result)
  def generate(fmt,text):
   prompt=''.join(render_parts(data,fmt,text));ids=tok(prompt,add_special_tokens=False,return_tensors='pt')['input_ids'].to(model.device)
   if ids.shape[1]>1024:raise RuntimeError('Frozen 1024-token feasibility ceiling exceeded; no truncation permitted')
   with torch.inference_mode():out=model.generate(input_ids=ids,do_sample=False,max_new_tokens=32,use_cache=True,pad_token_id=tok.eos_token_id,logits_to_keep=1)
   tokens=out[0,ids.shape[1]:].tolist();response=tok.decode(tokens,skip_special_tokens=True);strict,official=predictions('sst2',response)
   return dict(response=response,generated_ids=tokens,strict=strict,official=official,input_token_sha256=hashlib.sha256(json.dumps(ids[0].tolist()).encode()).hexdigest(),prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),input_tokens=ids.shape[1])
  for fmt in FORMATS:
   result['instruction_only'][fmt]=generate(fmt,'');save(path,result)
   for row in data['development']:
    result['development'].append(dict(format=fmt,source_id=row['source_id'],gold=row['label'],**generate(fmt,row['text'])));save(path,result)
   print('DEVELOPMENT',fmt,len(result['development']),flush=True)
  metrics={fmt:score([r for r in result['development'] if r['format']==fmt],result['instruction_only'][fmt]['strict']) for fmt in FORMATS}
  result['development_metrics']=metrics
  eligible=[fmt for fmt in FORMATS if metrics[fmt]['passed']]
  if eligible:
   winner=max(eligible,key=lambda f:(metrics[f]['strict_correct'],metrics[f]['strict_valid'],-FORMATS.index(f)))
   selection={'format':winner,'development_metrics':metrics[winner],'chosen_before_confirmation':True,'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
   save(OUT/'selected_configuration.json',selection);result['selection']=selection;save(path,result)
   # The selected format is now sealed. No second format is tried on confirmation.
   for row in data['screen']:
    result['confirmation'].append(dict(source_id=row['source_id'],gold=row['label'],**generate(winner,row['text'])));save(path,result)
   result['confirmation_metrics']=score(result['confirmation'],result['instruction_only'][winner]['strict'])
   result['status']='competence_passed' if result['confirmation_metrics']['passed'] else 'stopped_confirmation_gate'
  else:result['status']='stopped_no_development_candidate'
  result['backbone_after']=core.model_fingerprint(model);assert result['backbone_after']==result['backbone_before']
 except Exception as e:
  result.update(status='failed',error=str(e),traceback=traceback.format_exc());save(path,result);raise
 save(path,result);print('SCREEN_STATUS',result['status'],flush=True)
if __name__=='__main__':main()
