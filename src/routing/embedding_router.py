"""Fixed representations; inference accepts only trusted text, note, and question."""
import hashlib,re,time
import numpy as np
from scipy.sparse import csr_matrix

ENCODER='sentence-transformers/all-MiniLM-L6-v2'
REVISION='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'


def inputs(record):return record['trusted'],record['note'],record['question']


def lexical(rows,dimension=8192):
    """Signed hashed words/bigrams over the same three observable text fields."""
    indptr=[0];indices=[];data=[]
    for r in rows:
        counts={}
        for field,text in zip(['trusted','note','question'],inputs(r)):
            words=re.findall(r"[a-z0-9]+(?:'[a-z]+)?",text.casefold())
            grams=words+[' '.join(p) for p in zip(words,words[1:])]
            for gram in grams:
                h=int.from_bytes(hashlib.sha256((field+':'+gram).encode()).digest()[:8],'little')
                k=h%dimension;counts[k]=counts.get(k,0)+(1 if h>>63 else -1)
        norm=max(sum(x*x for x in counts.values())**.5,1.)
        for k,v in sorted(counts.items()):indices.append(k);data.append(v/norm)
        indptr.append(len(indices))
    return csr_matrix((data,indices,indptr),shape=(len(rows),dimension),dtype=np.float64)


def symbolic(rows):
    import adaptive_memory_router as routing
    return np.asarray([routing.features(*inputs(r),detailed=True)[0] for r in rows],dtype=np.float64)


class Encoder:
    def __init__(self):
        import torch
        from transformers import AutoModel,AutoTokenizer
        self.torch=torch;torch.set_num_threads(4)
        start=time.perf_counter()
        self.tokenizer=AutoTokenizer.from_pretrained(ENCODER,revision=REVISION)
        self.model=AutoModel.from_pretrained(ENCODER,revision=REVISION,use_safetensors=True).eval().to('cpu')
        self.model.requires_grad_(False);self.cache={}
        self.stats=dict(model=ENCODER,revision=REVISION,device='cpu',parameters=sum(p.numel() for p in self.model.parameters()),
                        loading_seconds=time.perf_counter()-start,encoding_seconds=0.,unique_texts=0,windows=0,max_text_tokens=0,
                        window_tokens=254,stride=192,maximum_sequence_with_special_tokens=256,batch_size=32)
    def populate(self,texts):
        unique=list(dict.fromkeys(t for t in texts if t not in self.cache))
        if not unique:return
        start=time.perf_counter();windows=[];owners=[];counts=[]
        for i,text in enumerate(unique):
            ids=self.tokenizer.encode(text,add_special_tokens=False,truncation=False)
            self.stats['max_text_tokens']=max(self.stats['max_text_tokens'],len(ids))
            chunks=[]
            for left in range(0,max(len(ids),1),192):
                chunks.append(ids[left:left+254])
                if left+254>=len(ids):break
            counts.append(len(chunks))
            for chunk in chunks:
                windows.append([self.tokenizer.cls_token_id, *chunk, self.tokenizer.sep_token_id]);owners.append(i)
        vectors=[]
        with self.torch.inference_mode():
            for begin in range(0,len(windows),32):
                batch=self.tokenizer.pad({'input_ids':windows[begin:begin+32]},padding=True,return_tensors='pt')
                output=self.model(**batch).last_hidden_state
                mask=batch['attention_mask'].unsqueeze(-1).to(output.dtype)
                pooled=(output*mask).sum(1)/mask.sum(1).clamp(min=1.)
                pooled=self.torch.nn.functional.normalize(pooled,p=2,dim=1)
                vectors.extend(pooled.cpu().numpy())
                if begin%640==0:print('ENCODE',begin,'/',len(windows),flush=True)
        by_owner=[[] for _ in unique]
        for owner,vec in zip(owners,vectors):by_owner[owner].append(vec)
        for text,vecs in zip(unique,by_owner):
            arr=np.stack(vecs);self.cache[text]=(arr.mean(0),arr.max(0))
        self.stats['encoding_seconds']+=time.perf_counter()-start
        self.stats['unique_texts']+=len(unique);self.stats['windows']+=len(windows)
    def transform(self,rows):
        self.populate([text for r in rows for text in inputs(r)])
        result=[]
        for r in rows:
            trusted,note,question=inputs(r)
            c=self.cache[trusted][0];m,mx=self.cache[note];q=self.cache[question][0]
            result.append(np.concatenate([c,q,m,mx,np.abs(c-m),c*m,np.abs(q-m),q*m]))
        return np.stack(result).astype(np.float64)
