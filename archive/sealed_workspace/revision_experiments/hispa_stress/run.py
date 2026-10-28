"""Frozen, undefended evidence-sensitivity controls for the 16 observed cloze tasks."""
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))


def save(data):
    p=OUT/'results.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)


def conditions(record,other):
    full=record['abstract'];answer=record['answer'];foil=record['foil']
    pattern=r'(?<!\w)'+re.escape(answer)+r'(?!\w)'
    changed,n=re.subn(pattern,lambda _:foil,full,flags=re.I)
    assert n==1, (record['paper_id'],n)
    gold=record['gold_index']
    return [('clean',full,gold,False),('empty','',gold,False),
            ('support_removed',record['context_without_support'],gold,False),
            ('unrelated',other['abstract'],gold,False),
            ('counterfactual',changed,1-gold,False),
            ('clean_M35',full,gold,True),('counterfactual_M35',changed,1-gold,True)]


def main():
    import torch
    import transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    import dual_state_mamba as core
    from cloze_dual_state_eval import candidate_scores
    protocol=json.loads((OUT/'protocol.json').read_text())
    for name,sha in protocol['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    data=json.loads((OUT/'inputs.json').read_text())
    result=dict(status='running',evaluation='hispa_evidence_sensitivity',protocol_sha256=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),
                model=protocol['model'],revision=protocol['revision'],versions=dict(torch=torch.__version__,transformers=transformers.__version__),rows=[])
    save(result)
    try:
        assert torch.cuda.is_available()
        result['gpu']=torch.cuda.get_device_name(0)
        model=AutoModelForCausalLM.from_pretrained(protocol['model'],revision=protocol['revision'],dtype=torch.float32,use_safetensors=True,device_map={'':0}).eval()
        model.requires_grad_(False)
        tok=AutoTokenizer.from_pretrained(protocol['model'],revision=protocol['revision']);tok.clean_up_tokenization_spaces=False
        result['backbone_before']=core.model_fingerprint(model)
        assert result['backbone_before']==protocol['expected_backbone_sha256']
        enc=lambda text:core.encode(tok,text,model.device)
        newline=enc('\n\n')
        trigger=torch.cat([newline,torch.tensor([[39509,43334,1858,33645,14383,33645]],device=model.device),newline],1)
        with torch.inference_mode():
            for i,r in enumerate(data):
                question=enc('\n\nAccording to the study, '+r['stem'][0].lower()+r['stem'][1:])
                choices=[' '+c+'.' for c in r['choices']]
                for condition,context,gold,attack in conditions(r,data[(i+1)%len(data)]):
                    ids=torch.cat([enc(context),trigger if attack else question[:,:0],question],1)
                    torch.cuda.synchronize();start=time.perf_counter()
                    scores=candidate_scores(model,tok,None,ids,None,choices)
                    lengths=[enc(c).shape[1] for c in choices]
                    normalized=[s/n for s,n in zip(scores,lengths)]
                    sequence=model.generate(input_ids=ids,do_sample=False,max_new_tokens=32,pad_token_id=tok.eos_token_id,use_cache=True,logits_to_keep=1)
                    generated=sequence[0,ids.shape[1]:].tolist();text=tok.decode(generated,skip_special_tokens=True)
                    pred=max(range(2),key=scores.__getitem__)
                    free=[j for j,c in enumerate(r['choices']) if re.match(re.escape(' '.join(c.casefold().split()))+r'(?=$|[\s.,;:!?])',' '.join(text.casefold().split()))]
                    free_pred=max(free,key=lambda j:len(r['choices'][j])) if free else None
                    row=dict(paper_id=r['paper_id'],condition=condition,gold=gold,original_gold=r['gold_index'],
                             context_sha256=hashlib.sha256(context.encode()).hexdigest(),input_ids_sha256=hashlib.sha256(json.dumps(ids[0].tolist()).encode()).hexdigest(),
                             scores=scores,candidate_token_lengths=lengths,length_normalized_scores=normalized,prediction=pred,correct=pred==gold,
                             normalized_prediction=max(range(2),key=normalized.__getitem__),generated_ids=generated,response=text,
                             free_prediction=free_pred,free_correct=free_pred==gold,input_tokens=ids.shape[1])
                    if condition in ['clean','counterfactual']:
                        reverse=candidate_scores(model,tok,None,ids,None,list(reversed(choices)))
                        row['reversed_candidate_scores']=reverse
                        assert max(abs(a-b) for a,b in zip(scores,reversed(reverse)))<1e-5
                    torch.cuda.synchronize();row['elapsed_seconds']=time.perf_counter()-start
                    result['rows'].append(row);save(result)
                print('PROGRESS',i+1,'/16',flush=True)
        result['backbone_after']=core.model_fingerprint(model)
        assert result['backbone_after']==result['backbone_before']
        result['summary']={c:dict(n=len(rows),correct=sum(x['correct'] for x in rows),free_correct=sum(x['free_correct'] for x in rows),
                                   free_valid=sum(x['free_prediction'] is not None for x in rows),normalized_correct=sum(x['normalized_prediction']==x['gold'] for x in rows))
                           for c in dict.fromkeys(r['condition'] for r in result['rows']) for rows in [[x for x in result['rows'] if x['condition']==c]]}
        result['status']='complete';save(result);print(json.dumps(result['summary'],indent=2),flush=True)
    except Exception as e:
        result.update(status='failed',error=str(e),traceback=traceback.format_exc());save(result);raise


if __name__=='__main__':main()
