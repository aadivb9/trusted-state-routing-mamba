"""Recompute historical candidate outcomes from saved likelihoods and sources."""
import hashlib
import json
from pathlib import Path
import random
import re

ROOT=Path(__file__).resolve().parent.parent


def read(name):
    return json.loads((ROOT/name).read_text())


def main():
    counts={}
    for name,data_path in [('text_router_benchmark','text_router_benchmark/dataset.json'),
                           ('adaptive_router_experiment','adaptive_router_experiment/independent_benchmark/dataset.json')]:
        result=read(name+'/results.json')
        source={r['id']:r for r in read(data_path)['records']}
        assert len(result['rows'])==600
        assert result['manifest']['backbone_before_sha256']==result['backbone_after_sha256']
        for row in result['rows']:
            spec=source[row['id']]['spec']
            index=max(range(len(row['log_likelihoods'])),key=lambda i:row['log_likelihoods'][i])
            prediction=spec['values'][index]
            assert row['prediction']==prediction and row['gold']==spec['actual']
            assert row['correct']==(prediction==spec['actual'])
            patterns={'documents':r'(\d+)\b','runs':r'(\d+)\b',
                      'hardware':r'(CPU|GPU)\b','method':r'(?:Method\s+)?([ABC])\b'}
            match=re.match(patterns[spec['slot']],row['generated_text'].lstrip(),re.I)
            free=match.group(1).upper() if match else None
            assert row['free_prediction']==free
            assert row['free_correct']==(free==spec['actual'])
        for variant in ['original','dual_state']:
            for condition in ['clean','safe','malicious']:
                rows=[r for r in result['rows'] if r['variant']==variant and r['condition']==condition]
                summary=result['summary'][variant][condition]['candidate']
                assert len(rows)==summary['n']==100
                assert sum(r['correct'] for r in rows)==summary['correct']
                assert summary['accuracy']==summary['correct']/100
                free=result['summary'][variant][condition]['free_generation']
                assert free['correct']==sum(r['free_correct'] for r in rows)
                assert free['n']==100 and free['accuracy']==free['correct']/100
        for condition in ['clean','safe','malicious']:
            left={r['group_id']:r['correct'] for r in result['rows'] if r['variant']=='original' and r['condition']==condition}
            right={r['group_id']:r['correct'] for r in result['rows'] if r['variant']=='dual_state' and r['condition']==condition}
            assert left.keys()==right.keys()
            delta=[int(right[k])-int(left[k]) for k in sorted(left)]
            paired=result['paired_differences'][condition]
            if 'correct' in paired:paired=paired['correct']
            assert paired['difference']==sum(delta)/len(delta)
            if name=='adaptive_router_experiment' and condition=='malicious':
                rng=random.Random(2026092602)
                samples=sorted(sum(rng.choices(delta,k=100))/100 for _ in range(5000))
                assert [samples[125],samples[4875]]==result['paired_differences'][condition]['bootstrap_95']
        counts[name]=dict(candidate_predictions_recomputed=600,free_predictions_recomputed=600,summary_cells_recomputed=12,paired_means_recomputed=3)
    result=read('generalization_experiment/results.json')
    source={r['paper_id']:r for r in read('generalization_experiment/dataset.json')['records']}
    rows=result['original']+result['dual']
    assert len(rows)==320
    for row in rows:
        pred=max(range(len(row['log_likelihoods'])),key=lambda i:row['log_likelihoods'][i])
        assert row['prediction']==pred and row['gold']==source[row['paper_id']]['gold_index']
        assert row['correct']==(pred==row['gold'])
        choices=source[row['paper_id']]['choices']
        text=' '.join(row['generated_text'].casefold().split())
        matches=[]
        for i,candidate in enumerate(choices):
            normalized=' '.join(candidate.casefold().split())
            if text.startswith(normalized) and (len(text)==len(normalized) or text[len(normalized)] in ' \t\n.,;:!?'):
                matches.append(i)
        free=max(matches,key=lambda i:len(choices[i])) if matches else None
        assert row['free_prediction']==free and row['free_correct']==(free==row['gold'])
    for role,records in [('original',result['original']),('dual_state',result['dual'])]:
        for condition in ['clean','safe_addition','attack_early','attack_late','mixed_note']:
            selected=[r for r in records if r['condition']==condition]
            for metric,key in [('candidate','correct'),('free_generation','free_correct')]:
                s=result['summary'][role][condition][metric]
                assert s['n']==len(selected)==32
                assert s['correct']==sum(r[key] for r in selected)
    filters=read('text_router_benchmark/filtering_equivalence.json')
    assert len(filters)==200
    assert all(r['candidate_prediction_matches'] and r['full_generated_text_matches'] for r in filters)
    assert len(result['filtering_checks'])==128 and len(result['mechanism_probes'])==4
    assert all(r['prediction_matches'] and r['generated_tokens_match'] for r in result['filtering_checks']+result['mechanism_probes'])
    counts['generalization']=dict(candidate_predictions_recomputed=320,free_predictions_recomputed=320,summary_cells_recomputed=20,filtering_checks_present=128,forced_probes_present=4)
    report=dict(status='passed',counts=counts,
                adaptive_headline_interval_reproduced=[.42,.62],text_filtering_matches=200,
                scope='Saved-evidence reanalysis; generation strings and recorded filtering booleans are not fresh GPU reruns.')
    (ROOT/'research_closeout/existing_results_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
