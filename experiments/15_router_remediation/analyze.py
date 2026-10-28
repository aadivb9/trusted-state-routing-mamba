"""Post-audit calculations on observed sets; no fitting or language-model inference."""
from collections import Counter, defaultdict
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'archive/sealed_workspace'
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.routing import context_router as historical
from src.routing.guarded_router import route, VERSION


def read(name):return json.loads((BASE/name).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def paired(rows, key):
    left={r['group_id']:bool(r[key]) for r in rows if r['variant']=='original' and r['condition']=='malicious'}
    right={r['group_id']:bool(r[key]) for r in rows if r['variant']=='dual_state' and r['condition']=='malicious'}
    assert set(left)==set(right) and len(left)==100
    values=[int(right[k])-int(left[k]) for k in sorted(left)]
    rng=random.Random(2026092602)
    draws=sorted(sum(rng.choices(values,k=len(values)))/len(values) for _ in range(5000))
    return {'n_source_groups':100,'difference':sum(values)/100,'ci95':[draws[125],draws[4875]],
            'pair_outcomes':dict(Counter(f'{int(left[k])}->{int(right[k])}' for k in left)),
            'scope':'Paired source-group bootstrap on the observed synthetic benchmark; conditional on its schema/provider/templates.'}


def counts(rows):
    summary={}
    for label in ('safe','malicious'):
        values=[r for r in rows if r['label']==label]
        c=Counter(r['action'] for r in values)
        summary[label]={'n':len(values),'accept':c['accept'],'reject':c['reject'],'abstain':c['abstain'],
                        'updates_blocked':c['reject']+c['abstain']}
    return summary


def main():
    spec=importlib.util.spec_from_file_location('independent_review',ROOT/'audit_review/recompute.py')
    review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)
    review.HERE=OUT
    with contextlib.redirect_stdout(io.StringIO()):review.main()
    audit=json.loads((OUT/'independent_checks.json').read_text())
    records=read('adaptive_router_experiment/independent_benchmark/dataset.json')['records']
    router=read('adaptive_router_experiment/router.json')
    results=read('adaptive_router_experiment/results.json')
    decisions=[];summary={}
    for name in ('cascade','fast_only','only_conflict_rule','guarded_v1'):
        rows=[]
        for r in records:
            trusted,note,question=r['trusted_context'],r['span'],r['spec']['stem']
            if name=='cascade':action='reject' if historical.route(router,trusted,note,question)['rejected'] else 'accept'
            elif name=='fast_only':action='reject' if historical.probability(router['fast'],historical.features(trusted,note,question)[0])>=router['threshold'] else 'accept'
            elif name=='only_conflict_rule':action='reject' if historical.features(trusted,note,question,detailed=True)[0][4] else 'accept'
            else:action=route(trusted,note,question)['action']
            rows.append({'id':r['id'],'label':r['attack_label'],'action':action})
        summary[name]=counts(rows)
        decisions.extend(dict(model=name,**r) for r in rows)
    original={r['id']:r['action'] for r in decisions if r['model']=='cascade'}
    disagreement={name:sum(r['action']!=original[r['id']] for r in decisions if r['model']==name) for name in summary}
    probes=[dict(name=p['name'],historical=p['decision'],revised=route(p['trusted'],p['note'],p['question'])) for p in audit['router_probes']]
    assert all(p['revised']['action']!='accept' for p in probes)
    coverage=defaultdict(Counter)
    for role,file in [('target','clean.json'),('injected','injected_baselines.json'),('attack','attacks.json')]:
        for r in read('external_benchmark/prepared/'+file):
            d=route(r['instruction'],r.get('text',r.get('untrusted_text')),r['instruction'])
            coverage[role][d['action']]+=1
    abstracts=[]
    for r in read('generalization_experiment/dataset.json')['records']:
        for field in ('benign_note','support_note','quoted_note','malicious_note','mixed_note'):
            abstracts.append({'paper_id':r['paper_id'],'note':field,**route(r['abstract'],r[field],r['stem'])})
    result={'status':'complete','scope':'Post-hoc observed-set CPU diagnostics only; no protected generation, fitting, or untouched-domain performance claim.',
            'code_sha256':sha(Path(__file__)),'router_sha256':sha(ROOT/'src/routing/guarded_router.py'),
            'source_audit_sha256':sha(ROOT/'audit_review/independent_checks.json'),
            'historical_input_sha256':audit['input_sha256'],'version':VERSION,
            'router_comparison':summary,'decision_disagreements_from_cascade':disagreement,'decisions':decisions,
            'paired_ranked':paired(results['rows'],'correct'),'paired_free':paired(results['rows'],'free_correct'),
            'audit_probes':probes,'external_observed_coverage':{k:dict(v) for k,v in coverage.items()},
            'abstract_observed_coverage':dict(Counter(r['action'] for r in abstracts)),
            'abstract_decisions':abstracts,'validation_calibration_non_embedding':audit['detector_transfer']['validation_calibration'],
            'sample_accounting':{'synthetic':{'source_groups':100,'notes':200,'attacked_pairs':100},
                                 'router_transfer':{'presentations':1050,'source_records':150,'source_groups':134,'models':6,'decisions':6300},
                                 'hispa_controls':{'reused_cases':16,'conditions':7,'evaluations':112},
                                 'instruction_gate':{'previously_observed_development_examples':72}}}
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','router_comparison','decision_disagreements_from_cascade','paired_free','external_observed_coverage','abstract_observed_coverage']},indent=2))


if __name__=='__main__':main()
