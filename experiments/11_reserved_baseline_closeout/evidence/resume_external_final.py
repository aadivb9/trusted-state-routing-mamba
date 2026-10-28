"""Resume only missing independent cases, with exact overlap-output controls."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from external_benchmark.run_competence import MODEL, REVISION, official_parsers, strict_label
from research_closeout.run_external_final import save


def run(output):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import dual_state_mamba as core
    output.mkdir(parents=True,exist_ok=False)
    original_seal=json.loads((ROOT/'research_closeout/seal.json').read_text())
    resume_seal=json.loads((ROOT/'research_closeout/resume_seal.json').read_text())
    for name,expected in {**original_seal['files'],**resume_seal['files']}.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    checkpoint=ROOT/'research_closeout/run_02/results.partial.json'
    result=json.loads(checkpoint.read_text())
    assert len(result['rows'])==572 and result['status']=='running'
    result['status']='running'
    result['resume']={'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      'resume_seal_sha256':hashlib.sha256((ROOT/'research_closeout/resume_seal.json').read_bytes()).hexdigest(),
                      'retained_rows':572,'missing_rows':28,'overlap_controls':[],
                      'instruction_controls':[], 'new_row_ids':[],
                      'weight_hash_scope':'Attempt 2 start and resumed segment start/end observed; attempt 2 terminal hash unavailable.'}
    path=output/'results.json'
    save(path,result)
    try:
        assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
        assert result['versions']=={'torch':torch.__version__,'transformers':transformers.__version__}
        model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,dtype=torch.float32,
                  device_map={'':0},low_cpu_mem_usage=True,use_safetensors=True).eval()
        model.requires_grad_(False)
        tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
        tokenizer.clean_up_tokenization_spaces=False
        before=core.model_fingerprint(model)
        assert before==result['backbone_before']
        result['resume']['backbone_before']=before
        # Retain the original function body, rather than hand-reimplement decoding.
        tree=ast.parse((ROOT/'research_closeout/run_external_final.py').read_text())
        function=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='generate')
        namespace={'torch':torch,'time':time,'tokenizer':tokenizer,'model':model}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'sealed_generate','exec'),namespace)
        generate=namespace['generate']
        parsers=official_parsers()
        examples={}
        order=[]
        for role,name in [('target','clean.json'),('injected','injected_baselines.json')]:
            for record in json.loads((ROOT/'external_benchmark/prepared'/name).read_text()):
                key=(role,record['dataset'],record['source_id'])
                examples[key]=record
                order.append(key)
        saved={(r['role'],r['dataset'],r['source_id']):r for r in result['rows']}
        assert len(saved)==572 and list(saved)==order[:572]
        with torch.inference_mode():
            for role in ['target','injected']:
                for task in ['sst2','sms_spam','mrpc']:
                    keys=[k for k in saved if k[:2]==(role,task)]
                    for key in [keys[0],keys[-1]]:
                        record=examples[key]
                        actual=generate(record['instruction']+'\nText: '+record['text'])
                        assert actual['generated_ids']==saved[key]['generated_ids'],f'Overlap mismatch: {key}'
                        result['resume']['overlap_controls'].append({'key':list(key),'generated_ids':actual['generated_ids'],'matched':True})
                        save(path,result)
            for key,control in result['instruction_only'].items():
                actual=generate(control['instruction']+'\nText: ')
                assert actual['generated_ids']==control['generated_ids'],'Instruction control mismatch'
                result['resume']['instruction_controls'].append({'key':key,'matched':True})
                save(path,result)
            print('All overlap and instruction controls matched',flush=True)
            for key in order[572:]:
                role,task,source_id=key
                record=examples[key]
                row=dict(role=role,dataset=task,source_id=source_id,gold=record['label'],
                         instruction_control=hashlib.sha256(record['instruction'].encode()).hexdigest(),
                         **generate(record['instruction']+'\nText: '+record['text']))
                row['strict_prediction']=strict_label(task,row['response'])
                row['official_prediction']=parsers[task](row['response'])
                row['strict_correct']=row['strict_prediction']==row['gold']
                row['official_correct']=row['official_prediction']==row['gold']
                result['rows'].append(row)
                result['resume']['new_row_ids'].append(list(key))
                save(path,result)
                print('PROGRESS',len(result['rows']),'/600',flush=True)
            for role in ['target','injected']:
                for task in ['sst2','sms_spam','mrpc']:
                    rows=[r for r in result['rows'] if r['role']==role and r['dataset']==task]
                    result['summary'][role+'/'+task]=dict(n=len(rows),strict_correct=sum(r['strict_correct'] for r in rows),
                        official_correct=sum(r['official_correct'] for r in rows),strict_valid=sum(r['strict_prediction'] is not None for r in rows))
            after=core.model_fingerprint(model)
            assert before==after
            result['resume']['backbone_after']=after
            result['backbone_after']=after
            result['backbone_unchanged']=before==after==result['backbone_before']
            result['resume']['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            result['resume']['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
            result['weight_hash_scope']=result['resume']['weight_hash_scope']
        result['status']='complete'
    except Exception as error:
        result.update(status='failed',error=str(error),traceback=traceback.format_exc())
        save(path,result)
        raise
    save(path,result)
    print(json.dumps(result['summary'],indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
