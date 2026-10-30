"""Deterministic observable-text segmentation and frozen encoder features."""
import re,time
import numpy as np

MODELS={
 'minilm':('sentence-transformers/all-MiniLM-L6-v2','1110a243fdf4706b3f48f1d95db1a4f5529b4d41','mean',''),
 'bge':('BAAI/bge-base-en-v1.5','a5beb1e3e68b9ab74eb54cfd186867f64f240e1a','cls',''),
 'e5':('intfloat/e5-base-v2','f52bf8ec8c7124536f0efb74aca902b2995e5bcd','mean','query: ')}

def segment(text,max_words=64):
 """Preserve exact bytes; boundaries depend only on observed text, never gold offsets."""
 cuts={0,len(text)}
 for m in re.finditer(r'\n+|(?<=[.!?])\s+',text):cuts.add(m.end())
 coarse=sorted(cuts);result=[]
 for left,right in zip(coarse,coarse[1:]):
  words=list(re.finditer(r'\S+',text[left:right]));starts=[left]+[left+words[i].start() for i in range(max_words,len(words),max_words)]+[right]
  result.extend({'left':a,'right':b,'text':text[a:b]} for a,b in zip(starts,starts[1:]) if b>a)
 assert ''.join(r['text'] for r in result)==text
 return result

class Encoder:
 def __init__(self,name,device='mps'):
  import torch
  from transformers import AutoModel,AutoTokenizer
  self.torch=torch;torch.set_num_threads(4);torch.manual_seed(20261029)
  model,rev,self.pool,self.prefix=MODELS[name];start=time.perf_counter()
  self.tokenizer=AutoTokenizer.from_pretrained(model,revision=rev)
  self.model=AutoModel.from_pretrained(model,revision=rev,use_safetensors=True,attn_implementation='eager').eval().to(device)
  self.model.requires_grad_(False);self.device=device;self.cache={}
  self.stats=dict(model=model,revision=rev,pooling=self.pool,prefix=self.prefix,device=device,parameters=sum(p.numel() for p in self.model.parameters()),load_seconds=time.perf_counter()-start,windows=0,texts=0,encoding_seconds=0.)
 def populate(self,texts):
  unique=list(dict.fromkeys(t for t in texts if t not in self.cache))
  if not unique:return
  start=time.perf_counter();prefix=self.tokenizer.encode(self.prefix,add_special_tokens=False);capacity=254-len(prefix);windows=[];owners=[]
  for i,text in enumerate(unique):
   ids=self.tokenizer.encode(text,add_special_tokens=False,truncation=False)
   for left in range(0,max(len(ids),1),192):
    body=ids[left:left+capacity];windows.append([self.tokenizer.cls_token_id,*prefix,*body,self.tokenizer.sep_token_id]);owners.append(i)
    if left+capacity>=len(ids):break
  pooled=[]
  with self.torch.inference_mode():
   for begin in range(0,len(windows),16):
    batch=self.tokenizer.pad({'input_ids':windows[begin:begin+16]},padding=True,return_tensors='pt').to(self.device)
    output=self.model(**batch).last_hidden_state
    mask=batch['attention_mask'].unsqueeze(-1).to(output.dtype)
    vec=output[:,0] if self.pool=='cls' else (output*mask).sum(1)/mask.sum(1).clamp(min=1.)
    vec=self.torch.nn.functional.normalize(vec,p=2,dim=1);pooled.extend(vec.cpu().numpy())
  groups=[[] for _ in unique]
  for owner,vec in zip(owners,pooled):groups[owner].append(vec)
  for text,vecs in zip(unique,groups):
   arr=np.stack(vecs);self.cache[text]=(arr.mean(0),arr.max(0))
  self.stats['windows']+=len(windows);self.stats['texts']+=len(unique);self.stats['encoding_seconds']+=time.perf_counter()-start
 def populate_records(self,records):
  texts=[]
  for r in records:texts.extend([r['trusted'],r['question']]);texts.extend(s['text'] for s in r['spans'])
  self.populate(texts)
 def feature(self,trusted,question,span_texts):
  c=self.cache[trusted][0];q=self.cache[question][0]
  means=np.stack([self.cache[t][0] for t in span_texts]);maxes=np.stack([self.cache[t][1] for t in span_texts]);m=means.mean(0);mx=maxes.max(0)
  return np.concatenate([c,q,m,mx,np.abs(c-m),c*m,np.abs(q-m),q*m]).astype(np.float64)
 def features(self,record,granularity):
  chunks=[s['text'] for s in record['spans']]
  if granularity=='document':return np.stack([self.feature(record['trusted'],record['question'],chunks)])
  if granularity=='span':return np.stack([self.feature(record['trusted'],record['question'],[s]) for s in chunks])
  raise ValueError('Unknown granularity')
