"""Prepare new source-disjoint records with the pinned published constructors."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'archive/sealed_workspace';OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(BASE),str(BASE/'external_benchmark')]
from external_benchmark.prepare_data import read_sources,identity,format_text,balanced,PublishedTask,ATTACKS,TASKS


def main():
    if (OUT/'prepared').exists():raise RuntimeError('Do not overwrite frozen selections')
    old=[]
    for name in ('development','clean','injected_baselines'):
        old.extend(json.loads((BASE/f'external_benchmark/prepared/{name}.json').read_text()))
    excluded={identity(r['text']) for r in old}
    used=set(excluded)
    sources,development_sources=read_sources()
    development_sources['sms_spam']=sources['sms_spam']
    instructions={name:next(r['instruction'] for r in old if r['dataset']==name) for name in TASKS}
    injected_old=json.loads((BASE/'external_benchmark/prepared/injected_baselines.json').read_text())
    injected_instructions={name:next(r['instruction'] for r in injected_old if r['dataset']==name) for name in TASKS}
    def choose(rows,n,salt):
        candidates=[];seen=set()
        for r in rows:
            key=identity(format_text(r['data']))
            if key not in used and key not in seen:candidates.append(r);seen.add(key)
        candidates.sort(key=lambda r:hashlib.sha256((salt+r['source_id']).encode()).hexdigest())
        result=balanced(candidates,n)
        used.update(identity(format_text(r['data'])) for r in result)
        return result
    dev=[];targets={};injected={};clean=[];inject_clean=[];attacks=[]
    for task in TASKS:
        dev += [dict(r,dataset=task,instruction=instructions[task],text=format_text(r['data'])) for r in choose(development_sources[task],24,'development-20261022')]
        targets[task]=choose(sources[task],100,'target-20261022')
        injected[task]=choose(sources[task],100,'injected-20261022')
        clean += [dict(r,dataset=task,instruction=instructions[task],text=format_text(r['data'])) for r in targets[task]]
        inject_clean += [dict(r,dataset=task,instruction=injected_instructions[task],text=format_text(r['data'])) for r in injected[task]]
    for target in TASKS:
        for attacker_task in TASKS:
            task=PublishedTask(injected[attacker_task],injected_instructions[attacker_task])
            for family,cls in ATTACKS.items():
                attack=cls(family,task)
                for i,r in enumerate(targets[target]):
                    attacks.append(dict(id=f'{target}/{attacker_task}/{family}/{i:03}',target_dataset=target,
                                        injected_dataset=attacker_task,attack=family,instruction=instructions[target],
                                        target_source_id=r['source_id'],injected_source_id=injected[attacker_task][i]['source_id'],
                                        target_label=r['label'],injected_label=task[i][1],untrusted_text=attack.inject(format_text(r['data']),i,TASKS[target])))
    dest=OUT/'prepared';dest.mkdir()
    for name,rows in [('development',dev),('clean',clean),('injected_baselines',inject_clean),('attacks',attacks)]:
        (dest/f'{name}.json').write_text(json.dumps(rows,indent=2)+'\n')
    report={'status':'prepared_not_evaluated','counts':{'development':len(dev),'clean':len(clean),'injected_baselines':len(inject_clean),'attacks':len(attacks)},
            'previously_observed_normalized_text_overlap':0,'development_final_normalized_text_overlap':0,
            'upstream_commit':json.loads((BASE/'external_benchmark/upstream_manifest.json').read_text())['commit'],
            'selection':'New fixed hash-order balanced selections with exact-normalized-text exclusion of historical and within-new selections; not the official first-N selection. No claim of semantic deduplication or unseen attack families.',
            'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir()}}
    groups=[{identity(r['text']) for r in rows} for rows in (dev,clean,inject_clean)]
    assert not any(g&excluded for g in groups)
    assert not any(groups[a]&groups[b] for a in range(3) for b in range(a))
    assert len(dev)==72 and len(attacks)==4500
    (OUT/'preparation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))


if __name__=='__main__':main()
