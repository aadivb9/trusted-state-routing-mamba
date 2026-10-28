"""One fixed instruction-tuned checkpoint on the previously observed development set."""
import hashlib,json,pathlib,sys,time,traceback
ROOT=pathlib.Path(__file__).resolve().parents[2];OUT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from external_benchmark.run_competence import strict_label,official_parsers


def save(d):
 p=OUT/'results.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)


def main():
 import torch,transformers
 from transformers import AutoTokenizer,AutoModelForCausalLM
 import dual_state_mamba as core
 p=json.loads((OUT/'protocol.json').read_text())
 for name,sha in p['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
 data=json.loads((ROOT/'external_benchmark/prepared/development.json').read_text())
 d=dict(status='running',evaluation='revision_instruction_checkpoint_development',model=p['model'],revision=p['revision'],
        protocol_sha256=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),rows=[],instruction_only={},
        versions=dict(torch=torch.__version__,transformers=transformers.__version__))
 save(d)
 try:
  assert torch.cuda.is_available();d['gpu']=torch.cuda.get_device_name(0)
  model=AutoModelForCausalLM.from_pretrained(p['model'],revision=p['revision'],dtype=torch.float32,use_safetensors=True,device_map={'':0}).eval()
  model.requires_grad_(False);tok=AutoTokenizer.from_pretrained(p['model'],revision=p['revision']);tok.clean_up_tokenization_spaces=False
  d['backbone_before']=core.model_fingerprint(model);parsers=official_parsers()
  def generate(instruction,text):
   # Original author's pinned chat template, supplied explicitly because the HF conversion omits it.
   prompt=tok.apply_chat_template([dict(role='user',content=instruction+'\nText: '+text)],chat_template=p['chat_template'],tokenize=False,add_generation_prompt=True)
   ids=tok(prompt,add_special_tokens=False,return_tensors='pt')['input_ids'].to(model.device)
   torch.cuda.synchronize();start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
   output=model.generate(input_ids=ids,do_sample=False,max_new_tokens=32,pad_token_id=tok.eos_token_id,use_cache=True,logits_to_keep=1)
   torch.cuda.synchronize();tokens=output[0,ids.shape[1]:].tolist()
   return dict(response=tok.decode(tokens,skip_special_tokens=True),generated_ids=tokens,input_tokens=ids.shape[1],
               prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),elapsed_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated())
  with torch.inference_mode():
   for task in ['sst2','sms_spam','mrpc']:
    group=[r for r in data if r['dataset']==task];assert len(group)==24
    instruction=group[0]['instruction'];d['instruction_only'][task]=generate(instruction,'');save(d)
    for r in group:
     row=dict(source_id=r['source_id'],dataset=task,gold=r['label'],**generate(instruction,r['text']))
     row.update(strict_prediction=strict_label(task,row['response']),official_prediction=parsers[task](row['response']))
     row.update(strict_correct=row['strict_prediction']==row['gold'],official_correct=row['official_prediction']==row['gold'])
     d['rows'].append(row);save(d);print('PROGRESS',len(d['rows']),'/72',flush=True)
  d['summary']={}
  for task in ['sst2','sms_spam','mrpc']:
   rows=[r for r in d['rows'] if r['dataset']==task];empty=strict_label(task,d['instruction_only'][task]['response'])
   correct=sum(r['strict_correct'] for r in rows);valid=sum(r['strict_prediction'] is not None for r in rows);gain=(correct-sum(r['gold']==empty for r in rows))/24
   d['summary'][task]=dict(n=24,strict_correct=correct,official_correct=sum(r['official_correct'] for r in rows),strict_valid=valid,input_gain=gain,passed=correct>=18 and valid>=22 and gain>=.2)
  d['gate_passed']=all(v['passed'] for v in d['summary'].values())
  d['backbone_after']=core.model_fingerprint(model);assert d['backbone_before']==d['backbone_after']
  d['status']='complete';save(d);print(json.dumps(d['summary'],indent=2),flush=True)
 except Exception as e:
  d.update(status='failed',error=str(e),traceback=traceback.format_exc());save(d);raise


if __name__=='__main__':main()
