"""Pinned reference/cache parity. CPU smoke is distinct from a T4 timing study."""
import argparse
import hashlib,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'archive/sealed_workspace')]
from src.runtime.pinned_models import load,PINS


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--model',choices=list(PINS),default='mamba_130m');args=parser.parse_args()
    output=OUT/('cpu_results.json' if args.device=='cpu' else 'gpu_results.json')
    if output.exists():raise RuntimeError('Preserve previous execution; use a new run version')
    d={'status':'running','device':args.device,'model':PINS[args.model][0],'revision':PINS[args.model][1],
       'protocol_sha256':hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),
       'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checks':[]}
    def save():output.write_text(json.dumps(d,indent=2)+'\n')
    save()
    try:
        import torch,transformers
        import dual_state_mamba as core
        torch.set_num_threads(4)
        d.update(versions={'torch':torch.__version__,'transformers':transformers.__version__},cuda_available=torch.cuda.is_available())
        if args.device=='cpu' and args.model!='mamba_130m':raise ValueError('CPU scope is the small-model reference smoke only')
        model,tok=load(args.model,require_cuda=args.device=='cuda')
        d['backbone_before']=core.model_fingerprint(model)
        ids=tok(('The study evaluated twenty-four documents on a CPU. '*12),add_special_tokens=False,return_tensors='pt')['input_ids'].to(model.device)
        reference={}
        with torch.inference_mode():
            for n in (8,16,32):
                reference[n]=model(input_ids=ids[:,:n+4],use_cache=False,logits_to_keep=1).logits.detach().clone()
            state=core.make_state_access(model,tok)
            for n in (8,16,32):
                captured=state.capture(ids[:,:n]);cache=captured['cache']
                backup=[t.clone() for t in state.tensors(cache)]
                candidate=state.clone(cache)
                independent=all(a.data_ptr()!=b.data_ptr() for a,b in zip(state.tensors(cache),state.tensors(candidate)))
                chunk=state.advance(ids[:,n:n+4],candidate).logits
                untouched=all(torch.equal(a,b) for a,b in zip(backup,state.tensors(cache)))
                token_cache=state.clone(cache)
                for j in range(n,n+4):single=state.advance(ids[:,j:j+1],token_cache).logits
                error=float((chunk-reference[n]).abs().max())
                token_error=float((chunk-single).abs().max())
                assert independent and untouched and error<=1e-3 and token_error<=1e-3
                d['checks'].append({'prefix_tokens':n,'suffix_tokens':4,'reference_max_error':error,
                                     'chunk_token_max_error':token_error,'independent_storage':independent,'trusted_state_unchanged':untouched})
                save()
        d['backbone_after']=core.model_fingerprint(model)
        assert d['backbone_before']==d['backbone_after']
        d.update(status='passed',scope='Fresh pinned small-model CPU reference/cache smoke; not a T4, large-model, throughput, training, or defense-efficacy result.' if args.device=='cpu' else 'Fresh reference/cache checks only; timing is a separate study.')
    except Exception as e:
        d.update(status='failed',error=str(e),traceback=traceback.format_exc())
        save();raise
    save();print(json.dumps(d,indent=2))


if __name__=='__main__':main()
