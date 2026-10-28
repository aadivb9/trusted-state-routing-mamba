"""Final frozen-router coverage; no language-model attack outcomes inferred."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from adaptive_memory_router import route


def main():
    router = json.loads((ROOT/'adaptive_router_experiment/router.json').read_text())
    rows = []
    for role, filename in [('target','clean.json'),('injected','injected_baselines.json'),('attack','attacks.json')]:
        source = json.loads((ROOT/'external_benchmark/prepared'/filename).read_text())
        for record in source:
            task = record.get('dataset', record.get('target_dataset'))
            text = record.get('text', record.get('untrusted_text'))
            decision = route(router,record['instruction'],text,record['instruction'])
            rows.append(dict(role=role, dataset=task,
                             id=record.get('id',record.get('source_id')),
                             **decision))
    summary={}
    for role in ['target','injected','attack']:
        for task in ['sst2','sms_spam','mrpc']:
            group=[r for r in rows if r['role']==role and r['dataset']==task]
            summary[role+'/'+task]=dict(n=len(group),rejected=sum(r['rejected'] for r in group),
                unknown_attribute=sum(r['evidence'].get('attribute') is None for r in group))
    result=dict(status='complete', scope='CPU routing only; no attack-conditioned generations',
                router_sha256=hashlib.sha256((ROOT/'adaptive_router_experiment/router.json').read_bytes()).hexdigest(),
                code_sha256=hashlib.sha256((ROOT/'adaptive_memory_router.py').read_bytes()).hexdigest(),
                summary=summary,rows=rows)
    (ROOT/'research_closeout/router_coverage.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
