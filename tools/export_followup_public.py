"""Explicit sanitized update; does not walk/copy local corpora or browser state."""
import hashlib,json,shutil
from pathlib import Path
import argparse
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--destination',type=Path,required=True);args=parser.parse_args()
SRC=Path(__file__).resolve().parents[1];DEST=args.destination.resolve()
if DEST==SRC or not (DEST/'provenance/public_manifest.json').is_file():raise SystemExit('Destination must be an existing separate public export')
summary=json.loads((SRC/'remediation/followup_20261029/summary.json').read_text())
if summary['external']['status'] not in ['complete','stopped_attack_validity_screen','failed']:raise SystemExit('Final follow-up evaluation is not terminal; do not export an interim state')
import subprocess,sys
subprocess.run([sys.executable,str(SRC/'tools/verify.py')],cwd=SRC,check=True)
sha=lambda raw:hashlib.sha256(raw).hexdigest()
origins_path=DEST/'provenance/export_origins.json';origins=json.loads(origins_path.read_text())
selected=['src/routing/embedding_followup.py','src/runtime/region_serialization.py',
'experiments/19_embedding_followup/prepare.py','experiments/19_embedding_followup/run.py','experiments/19_embedding_followup/replay_calibration.py','experiments/19_embedding_followup/protocol.json','experiments/19_embedding_followup/selection.json','experiments/19_embedding_followup/results.json','experiments/19_embedding_followup/frozen_heads.json','experiments/19_embedding_followup/calibration_replay.json','experiments/19_embedding_followup/raw/provenance.json',
'experiments/20_sst2_followup/prepare.py','experiments/20_sst2_followup/prompts.py','experiments/20_sst2_followup/run_screen.py','experiments/20_sst2_followup/prepare_masks.py','experiments/20_sst2_followup/run_final.py','experiments/20_sst2_followup/run_continuation.py','experiments/20_sst2_followup/continuation_amendment.json','experiments/20_sst2_followup/replay_masks.py','experiments/20_sst2_followup/mask_replay.json','experiments/20_sst2_followup/protocol.json','experiments/20_sst2_followup/final_protocol.json','experiments/20_sst2_followup/selected_configuration.json','experiments/20_sst2_followup/tokenization_audit.json',
'tools/export_followup_public.py','tools/prepare_public_followup_checks.py','tools/write_followup_public_docs.py','tools/verify_followup_public.py','paper/build.py','tools/followup_summary.py','tools/followup_report.py','tools/verify.py','tools/build_review_bundle.py','tools/run.py','tests/test_followup_round.py','tests/test_followup_evidence.py','tests/test_followup_serialization.py',
'paper/manuscript.md','paper/manuscript.pdf','paper/manuscript.html','paper/references.bib','paper/followup_results.md',
'remediation/followup_20261029/summary.json','remediation/followup_20261029/extra_normalization_probe.json','remediation/followup_20261029/AUTHOR_REVISION_PLAN.md','remediation/followup_20261029/reproduction_requirements.txt']
for name in selected:
 raw=(SRC/name).read_bytes();path=DEST/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
 origins['files'][name]={'source_path':name,'source_sha256':sha(raw),'export_sha256':sha(raw)}
s=json.loads((SRC/'remediation/followup_20261029/summary.json').read_text())
# Derived labels/scores and grouping only: no raw source text, questions, offsets or responses.
data=json.loads((SRC/'experiments/19_embedding_followup/results.json').read_text())
public_detector={k:{'threshold':data['heads'][k]['point_operating']['threshold'],'rows':[{name:r[name] for name in ['id','source_group','label','scores','rejected_spans','payload_spans_removed','payload_spans','benign_characters','benign_characters_removed']} for r in v['predictions']]} for k,v in data['final'].items()}
# IDs are benchmark identifiers, not private annotation IDs. Source hashes bind full evidence.
payload={'kind':'Derived follow-up observations; raw third-party text and model responses omitted','internal_detector_sha256':sha((SRC/'experiments/19_embedding_followup/results.json').read_bytes()),'internal_final_sha256':sha((SRC/'experiments/20_sst2_followup/final_results.json').read_bytes()),'detector':public_detector}
screen=json.loads((SRC/'experiments/20_sst2_followup/screen_results.json').read_text())
payload['screen']={k:screen[k] for k in ['status','development_metrics','confirmation_metrics','selection']}
payload['screen']['development']=[{k:r[k] for k in ['format','source_id','gold','strict','official']} for r in screen['development']]
payload['screen']['confirmation']=[{k:r[k] for k in ['source_id','gold','strict','official']} for r in screen['confirmation']]
payload['screen']['instruction_only']={k:{'strict':v['strict'],'official':v['official']} for k,v in screen['instruction_only'].items()}
final=json.loads((SRC/'experiments/20_sst2_followup/final_results.json').read_text())
payload['external']={k:final[k] for k in ['status','backbone_before','cache_checks','router_confidence_qualified']}
for k in ['backbone_after','attack_validity','error','continuation']:
 if k in final:payload['external'][k]=final[k]
keep=['id','role','source_id','family','gold','injected_gold','arm','strict','official','input_token_sha256','input_tokens','attack_strict_success','attack_published_success','injected_task_strict','injected_task_published']
payload['external']['rows']=[dict({k:r[k] for k in keep if k in r},generated_output_sha256=sha(json.dumps(r['generated_ids']).encode())) for r in final['rows']]
overlap=json.loads((SRC/'experiments/20_sst2_followup/continuation_overlap.json').read_text())
payload['overlap']=[dict({k:r[k] for k in keep if k in r},generated_output_sha256=sha(json.dumps(r['generated_ids']).encode())) for r in overlap['rows']]
path=DEST/'provenance/followup_observations.json';path.write_text(json.dumps(payload,indent=2)+'\n')
origins['source_snapshot']='Local research snapshot after October 29 bounded follow-up; earlier protocols retained'
origins['scope']='Sanitized first-party code, paper and derived observations. Full corpus/response reanalysis and exact sealed-code inference need the complete local evidence package. No inference repeated by exporting.'
origins_path.write_text(json.dumps(origins,indent=2)+'\n')
print(json.dumps({'copied_first_party_or_selected_records':len(selected),'derived_observations_bytes':path.stat().st_size}))
