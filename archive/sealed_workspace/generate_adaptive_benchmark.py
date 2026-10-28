"""Generate new Claude notes after freezing the adaptive router.

The old failed benchmark is a regression set, never relabeled as unseen data.
Generation sees controlled reference facts and styles, not router predictions.
"""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time

import generate_text_router_benchmark as source

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'adaptive_router_experiment'/'independent_benchmark'


def generate():
    OUT.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((OUT.parent/'protocol.json').read_text())
    for name,expected in frozen['frozen_files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    source.SEED=2026092702
    cases=source.source_cases()
    for index,case in enumerate(cases):
        old=case['study'];new=f'Nova-R{40000+index}'
        case['group_id']=f'adaptive_holdout_{index:04d}'
        for key,value in list(case.items()):
            if isinstance(value,str):case[key]=value.replace(old,new)
        case['spec']['stem']=case['spec']['stem'].replace(old,new)
    (OUT/'source_cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    target=OUT/'accepted_pairs.json'
    accepted=json.loads(target.read_text()) if target.exists() else {}
    journal=OUT/'generation_calls.jsonl'
    cost=sum(json.loads(x)['list_price_cost_usd'] for x in journal.read_text().splitlines()) if journal.exists() else 0.
    for start in range(0,100,20):
        pending=[c for c in cases[start:start+20] if c['group_id'] not in accepted]
        for attempt in range(3):
            if not pending:break
            if cost>=3:raise RuntimeError('Generation reached its $3 list-price ceiling.')
            prompt=source.prompt_for(pending)
            begin=time.perf_counter()
            command=['claude','-p','--safe-mode','--tools','','--no-session-persistence','--output-format','json',
                     '--model','sonnet','--effort','low','--max-budget-usd',str(min(1.5,3-cost)),
                     '--system-prompt','Produce the requested fictional research notes as JSON. No tools.']
            result=subprocess.run(command,input=prompt,text=True,capture_output=True,cwd='/tmp',timeout=240)
            response=json.loads(result.stdout) if result.stdout.strip().startswith('{') else {}
            call={'batch_start':start,'attempt':attempt,'prompt':prompt,'response':response,'stderr':result.stderr,
                  'returncode':result.returncode,'seconds':time.perf_counter()-begin,'list_price_cost_usd':response.get('total_cost_usd',0)}
            cost+=call['list_price_cost_usd']
            with journal.open('a') as handle:handle.write(json.dumps(call)+'\n')
            if result.returncode or response.get('is_error'):raise RuntimeError('Claude generation failed; raw call saved.')
            pairs={p['group_id']:p for p in source.parse_response(response['result'])}
            retry=[]
            for case in pending:
                pair=pairs.get(case['group_id'],{})
                errors=source.validate_pair(pair,case)
                if errors:
                    print('Structural retry',case['group_id'],errors,flush=True);retry.append(case)
                else:accepted[case['group_id']]=pair
            pending=retry
            target.write_text(json.dumps(accepted,indent=2)+'\n')
            print('Accepted',len(accepted),'/100 groups; reported list-price',round(cost,4),flush=True)
        if pending:raise RuntimeError('Unresolved structural errors; no performance-based selection allowed.')
    rows=[]
    seen=set()
    for case in cases:
        for label in ['safe','malicious']:
            span=accepted[case['group_id']][label+'_span']
            assert span not in seen;seen.add(span)
            rows.append({'id':case['group_id']+'_'+label,'group_id':case['group_id'],
                'trusted_context':case['safe_context'] if label=='safe' else case['full_context'],
                'span':span,'attack_label':label,'family':case['safe_style'] if label=='safe' else case['attack_style'],
                'requires_safe_span':case['requires_safe_span'] and label=='safe','spec':case['spec'],
                'target_fact':case['target_fact'],'split':'independent_test','generator':'Claude CLI sonnet; temperature unavailable'})
    (OUT/'dataset.json').write_text(json.dumps({'records':rows,'generation_cost_list_price':cost},indent=2)+'\n')
    print('Independent benchmark generated. Semantic audit must finish before evaluation.',flush=True)


if __name__=='__main__':generate()
