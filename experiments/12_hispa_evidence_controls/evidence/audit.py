"""Independently rescore saved stress outputs and retain comparison provenance."""
import hashlib,json,math,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent


def main():
 p=json.loads((OUT/'protocol.json').read_text());r=json.loads((OUT/'results.json').read_text())
 assert r['status']=='complete' and len(r['rows'])==112
 for name,h in p['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
 assert r['protocol_sha256']==hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()
 assert r['backbone_before']==r['backbone_after']==p['expected_backbone_sha256']
 records={x['paper_id']:x for x in json.loads((OUT/'inputs.json').read_text())}
 old=json.loads((ROOT/'final_hispa_audit/results.json').read_text());old={(x['paper_id'],x['condition']):x for x in old['rows']}
 assert len({(x['paper_id'],x['condition']) for x in r['rows']})==112
 comparison=[];reverse_checks=0
 for row in r['rows']:
  task=records[row['paper_id']];changed=row['condition'].startswith('counterfactual')
  assert row['gold']==(1-task['gold_index'] if changed else task['gold_index'])
  scores=row['scores'];pred=max(range(2),key=scores.__getitem__)
  assert pred==row['prediction'] and row['correct']==(pred==row['gold'])
  normalized=[s/n for s,n in zip(scores,row['candidate_token_lengths'])]
  assert normalized==row['length_normalized_scores']
  assert row['normalized_prediction']==max(range(2),key=normalized.__getitem__)
  text=' '.join(row['response'].casefold().split());matches=[]
  for j,c in enumerate(task['choices']):
   value=' '.join(c.casefold().split())
   if re.match(re.escape(value)+r'(?=$|[\s.,;:!?])',text):matches.append(j)
  free=max(matches,key=lambda j:len(task['choices'][j])) if matches else None
  assert row['free_prediction']==free and row['free_correct']==(free==row['gold'])
  assert 0<len(row['generated_ids'])<=32
  if 'reversed_candidate_scores' in row:
   assert max(abs(a-b) for a,b in zip(scores,reversed(row['reversed_candidate_scores'])))<1e-5;reverse_checks+=1
  if row['condition'] in ['clean','clean_M35','empty']:
   name={'clean':'clean','clean_M35':'M_35','empty':'empty_context'}[row['condition']]
   error=max(abs(a-b) for a,b in zip(scores,old[row['paper_id'],name]['original_scores']))
   comparison.append(error)
 summary={}
 for condition in p['conditions']:
  rows=[x for x in r['rows'] if x['condition']==condition];assert len(rows)==16
  summary[condition]=dict(n=16,correct=sum(x['correct'] for x in rows),free_correct=sum(x['free_correct'] for x in rows),
                          free_valid=sum(x['free_prediction'] is not None for x in rows),normalized_correct=sum(x['normalized_prediction']==x['gold'] for x in rows))
 assert summary==r['summary'];assert reverse_checks==32
 audit=dict(status='passed',rows_rescored=112,candidate_order_checks=reverse_checks,existing_score_comparisons=len(comparison),
            maximum_existing_score_error=max(comparison),all_existing_scores_within_1e_3=max(comparison)<1e-3,
            backbone_unchanged=True,result_sha256=hashlib.sha256((OUT/'results.json').read_bytes()).hexdigest(),
            scope='New inference on reused cases; saved scores independently rescored. Does not establish official HiSPA replication or absence of pretraining overlap.')
 (OUT/'verification.json').write_text(json.dumps(audit,indent=2)+'\n')
 lines=['# Evidence-sensitivity stress test','',
        'Completed on the same frozen FP32 Mamba-130M checkpoint and T4. All 16 previously observed pilot tasks were retained; 112 new case/condition evaluations completed. This is a diagnostic, not a new untouched benchmark.','',
        '| Condition | Candidate ranking | Strict free generation | Length-normalized ranking (secondary) |',
        '| --- | ---: | ---: | ---: |']
 for condition,s in summary.items():lines.append(f'| {condition} | {s["correct"]}/16 | {s["free_correct"]}/16 | {s["normalized_correct"]}/16 |')
 lines+=['','The artificial counterfactual replaces the unique answer in the supplied abstract with its foil and flips the expected answer. This tests responsiveness to altered evidence, not factual correctness of a real publication. Unrelated-source and removed-source conditions retain the original target solely to measure how often the original answer persists without its intended evidence.','',
         '## Interpretation','',
         'The original 100% is confined to two-choice ranking. Free generation produces only 7/16 valid correct opening answers under the strict parser on clean and published-trigger prompts. Format failures are mixed with semantic failures in that metric.','',
         'Ranking follows altered evidence in 15/16 counterfactual cases, so a fixed leaked correct-option index does not explain the observed behavior. However, 12/16 original answers survive with no source, 11/16 survive removal of the supporting sentence, and 10/16 survive an unrelated source. Strong priors and the literal-answer task shortcut remain material concerns. These interventions cannot exclude every form of leakage or pretraining overlap.','',
         'The optimized M35 trigger still leaves clean ranking at 16/16. On altered-source cases it yields 14/16 versus 15/16 without the trigger. This small post-hoc stress result has no matched benign counterfactual control and is not evidence of a reproduced HiSPA effect or successful defense. The new run evaluates the undefended model only.','',
         'Secondary length normalization leaves the main clean, empty, support-removal and counterfactual counts unchanged; it changes the unrelated and counterfactual-trigger results. The declared primary metric remains summed likelihood.','',
         f'All 112 saved rows were rescored, 32 reversed-candidate checks passed, and the maximum difference across 48 comparable historical score vectors was {max(comparison):.8g}. The before/after weight hashes match the historical checkpoint. These checks support reproducibility of the scoring path; they are not independent population replications.','',
         'Conclusion: the ceiling is an easy restricted-choice result with appreciable prior-driven accuracy, not a general robustness result. More realistic published task/distractor reproduction is still needed for an official HiSPA claim.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
