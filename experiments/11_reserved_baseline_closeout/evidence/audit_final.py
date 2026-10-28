"""Audit saved final diagnostic responses independently of the GPU evaluator."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from external_benchmark.audit_development import predictions


def main():
    seal_path=ROOT/'research_closeout/seal.json'
    seal=json.loads(seal_path.read_text())
    for name,expected in seal['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    path=ROOT/'research_closeout/run_03/results.json'
    result=json.loads(path.read_text())
    assert result['status']=='complete'
    resume=result['resume']
    resume_seal_path=ROOT/'research_closeout/resume_seal.json'
    resume_seal=json.loads(resume_seal_path.read_text())
    assert resume['resume_seal_sha256']==hashlib.sha256(resume_seal_path.read_bytes()).hexdigest()
    for name,expected in resume_seal['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    checkpoint_path=ROOT/'research_closeout/run_02/results.partial.json'
    checkpoint=json.loads(checkpoint_path.read_text())
    assert resume['checkpoint_sha256']==hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert result['rows'][:572]==checkpoint['rows']
    prior={(r['role'],r['dataset'],r['source_id']):r for r in checkpoint['rows']}
    assert len(resume['overlap_controls'])==12
    for control in resume['overlap_controls']:
        assert control['matched'] and control['generated_ids']==prior[tuple(control['key'])]['generated_ids']
    assert {x['key'] for x in resume['instruction_controls']}==set(result['instruction_only'])
    assert all(x['matched'] for x in resume['instruction_controls'])
    assert len(resume['new_row_ids'])==28
    assert [list((r['role'],r['dataset'],r['source_id'])) for r in result['rows'][572:]]==resume['new_row_ids']
    assert resume['backbone_before']==resume['backbone_after']==checkpoint['backbone_before']
    assert result['seal_sha256']==hashlib.sha256(seal_path.read_bytes()).hexdigest()
    assert result['attack_generations_evaluated']==0
    source={}
    for role,name in [('target','clean.json'),('injected','injected_baselines.json')]:
        for record in json.loads((ROOT/'external_benchmark/prepared'/name).read_text()):
            source[role,record['dataset'],record['source_id']]=record
    assert len(source)==len(result['rows'])==600
    seen=set()
    for row in result['rows']:
        key=row['role'],row['dataset'],row['source_id']
        assert key not in seen
        seen.add(key)
        assert row['gold']==source[key]['label']
        strict,official=predictions(row['dataset'],row['response'])
        assert (strict,official)==(row['strict_prediction'],row['official_prediction'])
        assert row['strict_correct']==(strict==row['gold'])
        assert row['official_correct']==(official==row['gold'])
        assert 0<len(row['generated_ids'])<=32
        control=result['instruction_only'][row['instruction_control']]
        assert control['instruction']==source[key]['instruction']
    for key,summary in result['summary'].items():
        role,task=key.split('/')
        rows=[r for r in result['rows'] if r['role']==role and r['dataset']==task]
        assert len(rows)==100 and sum(r['gold'] for r in rows)==50
        assert summary==dict(n=100,strict_correct=sum(r['strict_correct'] for r in rows),
                             official_correct=sum(r['official_correct'] for r in rows),
                             strict_valid=sum(r['strict_prediction'] is not None for r in rows))
    assert len(result['summary'])==6
    hardware=json.loads((ROOT/'mamba_28b_feasibility/fp32_run_02/results.json').read_text())
    assert result['backbone_before']==result['backbone_after']==hardware['backbone_before']
    partial_path=ROOT/'research_closeout/run_01/results.partial.json'
    recovery={}
    if partial_path.exists():
        partial=json.loads(partial_path.read_text())
        current={(r['role'],r['dataset'],r['source_id']):r for r in result['rows']}
        previous=partial['rows']
        recovery=dict(previous_rows=len(previous),
                      identical_generated_token_sequences=sum(r['generated_ids']==current[r['role'],r['dataset'],r['source_id']]['generated_ids'] for r in previous),
                      interpretation='Computational repeat of the same sealed inputs; not independent samples or a pooled run.')
    coverage=json.loads((ROOT/'research_closeout/router_coverage.json').read_text())
    assert len(coverage['rows'])==5100
    assert len({(r['role'],r['dataset'],r['id']) for r in coverage['rows']})==5100
    for key,summary in coverage['summary'].items():
        role,task=key.split('/')
        rows=[r for r in coverage['rows'] if r['role']==role and r['dataset']==task]
        assert summary==dict(n=len(rows),rejected=sum(r['rejected'] for r in rows),
                             unknown_attribute=sum(r['evidence'].get('attribute') is None for r in rows))
    audit=dict(status='passed',responses_recomputed=600,summary_cells_recomputed=6,
               sealed_files_verified=len(seal['files']),router_records_checked=5100,
               backbone_hash_matches_hardware_trial=True,
               recovery_comparison=recovery,
               retained_checkpoint_rows=572,new_continuation_rows=28,exact_overlap_controls=12,
               interrupted_segment_terminal_hash_available=False,
               result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               scope='Saved-response scores and observed boundary hashes; interrupted-segment terminal hash unavailable. Overlap controls are computational repeats, not new independent samples.')
    (ROOT/'research_closeout/run_03/verification.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
