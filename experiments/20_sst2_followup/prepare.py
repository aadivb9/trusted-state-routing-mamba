"""New SST-2 development/screen/final selections; no model outcomes used."""
import hashlib,json,math,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'archive/sealed_workspace'),str(ROOT/'archive/sealed_workspace/external_benchmark')]
from external_benchmark.prepare_data import read_sources,identity,format_text,balanced,PublishedTask,ATTACKS,TASKS

def key(s):return hashlib.sha256(s.encode()).hexdigest()
def shingles(s):
 w=identity(s).split();return set(zip(*(w[i:] for i in range(5))))

def main():
 assert not (OUT/'prepared.json').exists(),'Preserve previous selections'
 source,dev=read_sources();seen=set();prior=[]
 for base in [ROOT/'archive/sealed_workspace/external_benchmark/prepared',ROOT/'experiments/17_external_validation/prepared']:
  for name in ['development','clean','injected_baselines']:
   for r in json.loads((base/(name+'.json')).read_text()):seen.add(identity(r['text']));prior.append(r['text'])
 refs=[shingles(t) for t in prior]
 def choose(rows,n,role):
  selected=[];counts={0:0,1:0}
  for r in sorted(rows,key=lambda r:key('20261029/'+role+'/'+r['source_id'])):
   text=format_text(r['data']);normal=identity(text);sh=shingles(text)
   if normal in seen or counts[r['label']]>=n//2:continue
   if sh and any(len(sh&old)/len(sh|old)>=.9 for old in refs if old):continue
   selected.append(dict(source_id=r['source_id'],text=text,label=r['label']));seen.add(normal);refs.append(sh);counts[r['label']]+=1
   if len(selected)==n:break
  assert len(selected)==n,(role,len(selected))
  return selected
 demos=choose(dev['sst2'],4,'demonstrations');development=choose(dev['sst2'],40,'prompt_development')
 screen=choose(dev['sst2'],100,'confirmation_screen');targets=choose(source['sst2'],50,'final_targets')
 injections=choose(source['sms_spam'],50,'attacker_spam')
 folder=ROOT/'archive/sealed_workspace/external_benchmark/upstream/data/system_prompts'
 instruction=(folder/'sentiment_analysis.txt').read_text().splitlines()[0].rstrip()
 inject_instruction=(folder/'spam_detection_inject.txt').read_text().splitlines()[0].rstrip()
 task=PublishedTask([dict(r,data=r['text']) for r in injections],inject_instruction)
 attacks=[]
 for family,cls in ATTACKS.items():
  operator=cls(family,task)
  for i,r in enumerate(targets):
   attacks.append(dict(id=f'{family}/{i:03}',family=family,target_source_id=r['source_id'],gold=r['label'],injected_gold=injections[i]['label'],text=operator.inject(r['text'],i,TASKS['sst2'])))
 data=dict(instruction=instruction,injected_instruction=inject_instruction,demonstrations=demos,development=development,screen=screen,targets=targets,injections=injections,attacks=attacks)
 (OUT/'prepared.json').write_text(json.dumps(data,indent=2)+'\n')
 (OUT/'selection_audit.json').write_text(json.dumps({'selection':'Fixed hash-order balanced samples; normalized-exact and 5-word Jaccard >=0.90 exclusion from historical inputs and between new roles.','counts':{k:len(v) for k,v in data.items() if isinstance(v,list)},'historical_exclusion_records':len(prior),'prepared_sha256':hashlib.sha256((OUT/'prepared.json').read_bytes()).hexdigest(),'scope':'New development-informed one-task study; not a continuation or retroactive pass of the prior three-task protocol. No foundation-model pretraining independence claim.'},indent=2)+'\n')
 print('Prepared',len(attacks),'attacks; 40 prompt development and 100 confirmation examples.')
if __name__=='__main__':main()
