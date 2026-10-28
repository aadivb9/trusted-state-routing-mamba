"""Independently audit the fixed instruction-checkpoint competence screen."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from external_benchmark.audit_development import predictions


def main():
 p=json.loads((OUT/'protocol.json').read_text());r=json.loads((OUT/'results.json').read_text())
 assert r['status']=='complete' and len(r['rows'])==72
 for name,h in p['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
 assert r['protocol_sha256']==hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()
 assert r['backbone_before']==r['backbone_after']
 data={(x['dataset'],x['source_id']):x for x in json.loads((ROOT/'external_benchmark/prepared/development.json').read_text())}
 seen=set()
 for row in r['rows']:
  key=row['dataset'],row['source_id'];assert key not in seen;seen.add(key)
  assert row['gold']==data[key]['label']
  strict,official=predictions(row['dataset'],row['response'])
  assert (strict,official)==(row['strict_prediction'],row['official_prediction'])
  assert row['strict_correct']==(strict==row['gold']) and row['official_correct']==(official==row['gold'])
  assert 0<len(row['generated_ids'])<=32
 assert seen==set(data)
 summary={}
 for task in ['sst2','sms_spam','mrpc']:
  rows=[x for x in r['rows'] if x['dataset']==task];assert len(rows)==24 and sum(x['gold'] for x in rows)==12
  strict=sum(x['strict_correct'] for x in rows);valid=sum(x['strict_prediction'] is not None for x in rows)
  empty=predictions(task,r['instruction_only'][task]['response'])[0];gain=(strict-sum(x['gold']==empty for x in rows))/24
  summary[task]=dict(n=24,strict_correct=strict,official_correct=sum(x['official_correct'] for x in rows),strict_valid=valid,input_gain=gain,
                     passed=strict>=18 and valid>=22 and gain>=.2)
 assert summary==r['summary'];assert r['gate_passed']==all(v['passed'] for v in summary.values())
 audit=dict(status='passed',responses_recomputed=72,summary_cells_recomputed=3,gate_passed=r['gate_passed'],backbone_unchanged=True,
            result_sha256=hashlib.sha256((OUT/'results.json').read_bytes()).hexdigest(),
            scope='Previously observed development samples, fixed third-party converted checkpoint, author chat template and published task instructions. Not a final test or direct native-checkpoint equivalence validation.')
 (OUT/'verification.json').write_text(json.dumps(audit,indent=2)+'\n')
 lines=['# Instruction-tuned Mamba development screen','',
        'Completed all 72 previously observed development examples with one fixed FP32 instruction-tuned checkpoint and the original author chat template. No prompt sweep or task selection followed these outcomes.','',
        '| Task | Strict correct | Published scorer correct | Strict valid | Gate |',
        '| --- | ---: | ---: | ---: | --- |']
 for task,s in summary.items():lines.append(f'| {task} | {s["strict_correct"]}/24 | {s["official_correct"]}/24 | {s["strict_valid"]}/24 | {"Pass" if s["passed"] else "Fail"} |')
 lines+=['','All three tasks fail the unchanged requirement of at least 18 strict correct, 22 valid and 20 percentage-point gain over instruction-only accuracy. This configuration therefore does not advance to an end-to-end security-efficacy test. The separate router detection study remains valid within its own stated scope.','',
         'Many generated responses use explanatory prose instead of the requested label format. The substring scorer can also misread negation or echoed labels. These scores characterize the fixed prompt/checkpoint/parser setup, not an intrinsic limit on semantic knowledge. No post-hoc parser or prompt was substituted to convert the failed gate into a pass.','',
         'Checkpoint: `han1997/mamba-2.8b-zephyr-hf` at `99a87b5e5f211c42595df370de50910f05dc7854`, a third-party Transformers conversion of `xiuyul/mamba-2.8b-zephyr`. The conversion card states that weights are unchanged; this run does not independently validate native-to-HF conversion equivalence. The author chat template was fetched from original revision `85b6d9a1dd2bca5787437ba63af63f155356e113` because the converted tokenizer omits it.','',
         'The before/after parameter hashes match. All 72 saved scores and three gate decisions were independently recomputed. No backbone weights were trained. This was development on already observed examples, not unseen-domain evaluation.','',
         'Source cards: https://huggingface.co/han1997/mamba-2.8b-zephyr-hf ; https://huggingface.co/xiuyul/mamba-2.8b-zephyr']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
