"""Independent saved-prediction audit of all frozen router/domain cells."""
from collections import Counter
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def auc(y,p):
 positives=[v for a,v in zip(y,p) if a];negatives=[v for a,v in zip(y,p) if not a]
 return sum((a>b)+.5*(a==b) for a in positives for b in negatives)/(len(positives)*len(negatives))

def main():
 p=json.loads((OUT/'protocol.json').read_text());a=json.loads((OUT/'runtime_amendment.json').read_text())
 for name,h in p['files'].items():
  if name in a['files']:
   assert sha(ROOT/(name+'.before_tokenizer_fix'))==h
   assert sha(ROOT/name)==a['files'][name]
  else:assert sha(ROOT/name)==h,name
 fitted=json.loads((OUT/'fitted_models.json').read_text());seal=json.loads((OUT/'fitted_seal.json').read_text());result=json.loads((OUT/'results.json').read_text())
 assert result['status']=='complete'
 assert seal['models_sha256']==result['fitted_models_sha256']==sha(OUT/'fitted_models.json')
 assert seal['protocol_sha256']==result['protocol_sha256']==sha(OUT/'protocol.json')
 assert seal['runtime_amendment_sha256']==result['runtime_amendment_sha256']==sha(OUT/'runtime_amendment.json')
 prepared=json.loads((OUT/'prepared.json').read_text());final={r['id']:r for r in prepared if r['split']=='final'}
 assert len(final)==1050 and len(fitted)==6 and len(result['rows'])==6300
 expected={(key,id) for key in fitted for id in final};seen=set()
 for row in result['rows']:
  key=(row['model'],row['id']);assert key not in seen;seen.add(key)
  source=final[row['id']]
  for field in ['domain','source_group','label','position','attack_family']:assert row[field]==source[field]
  assert 0<=row['risk']<=1 and row['rejected']==(row['risk']>=fitted[row['model']]['threshold'])
 assert seen==expected
 for key,summary in result['summary'].items():
  model,domain=key.rsplit('/',1);rows=[r for r in result['rows'] if r['model']==model and r['domain']==domain]
  benign=[r for r in rows if r['label']==0];attack=[r for r in rows if r['label']==1]
  assert summary['n']==len(rows) and summary['source_groups']==len({r['source_group'] for r in rows})
  assert summary['true_positive']==sum(r['rejected'] for r in attack) and summary['false_positive']==sum(r['rejected'] for r in benign)
  assert abs(summary['recall']-sum(r['rejected'] for r in attack)/len(attack))<1e-12
  assert abs(summary['false_positive_rate']-sum(r['rejected'] for r in benign)/len(benign))<1e-12
  assert abs(summary['roc_auc']-auc([r['label'] for r in rows],[r['risk'] for r in rows]))<1e-12
  brier=sum((r['risk']-r['label'])**2 for r in rows)/len(rows)
  assert abs(summary['brier']-brier)<1e-12
 assert len(result['summary'])==12
 for first,second in [('train','validation'),('train','final'),('validation','final')]:
  assert not ({r['source_group'] for r in prepared if r['split']==first}&{r['source_group'] for r in prepared if r['split']==second})
 assert {r['domain'] for r in prepared if r['split']=='final'}=={'table','code'}
 assert {r['domain'] for r in prepared if r['split'] in ['train','validation']}=={'synthetic','email'}
 report=dict(status='passed',predictions_checked=6300,summary_cells_recomputed=12,models=6,final_presentations=1050,
             final_source_groups=134,train_validation_final_source_group_overlap=0,
             result_sha256=sha(OUT/'results.json'),scope='Saved predictions, score thresholds, source separation, AUROC and Brier independently checked; not a new encoder inference or end-to-end security test.')
 (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
 lines=['# Frozen-router domain transfer: embeddings and data diversity','',
        'Completed a fixed 3-representation × 2-training-corpus experiment. Table and code domains were excluded from router fitting and threshold selection. No model or threshold was revised after these final outcomes.','',
        '| Training corpus | Representation | Domain | Attacks rejected | Benign rejected | AUROC |',
        '| --- | --- | --- | ---: | ---: | ---: |']
 for key,s in result['summary'].items():
  regime,rep,domain=key.split('/')
  lines.append(f'| {regime} | {rep} | {domain} | {s["true_positive"]}/{s["attack_n"]} ({s["recall"]:.1%}) | {s["false_positive"]}/{s["benign_n"]} ({s["false_positive_rate"]:.1%}) | {s["roc_auc"]:.3f} |')
 lines+=['','## What the comparison establishes','',
         'Changing the representation alone did not produce reliable cross-domain detection. Synthetic-only embeddings rejected many legitimate table inputs. Adding independently sourced email training lowered that table false-positive rate but did not bring it close to the development target; code detection remained weak. The embedding model did not consistently outperform the symbolic representation on every domain.','',
         'The fixed operating point was chosen on a common mixed validation set to maximize recall subject to an aggregate benign false-positive rate of at most 5%. It is not a guarantee for each validation domain or any final domain. The validation set contains 100 synthetic and 10 email benign examples, so the aggregate constraint can conceal weaker email calibration. This is a protocol limitation, not a reason to recalibrate against the final results.','',
         'A reject decision here applies to an entire external document. Rejecting benign documents can destroy task utility. These are detector measurements only: no task-answer accuracy, attack-success reduction, or benefit unique to dual-state memory is established by this experiment.','',
         '## Design and provenance','',
         '- Same 1000 synthetic training examples in both regimes; the expanded regime adds 280 presentations from40 email records. Both use the same 200 synthetic plus 70 email validation presentations. Synthetic-only therefore means training-only; email is present during threshold selection for both regimes.',
         '- All heads use fixed C=1 logistic regression, balanced class weights and no hyperparameter sweep. Dense scaling is fitted only on training. The symbolic condition is a new single logistic head on 14 detailed features, not a rerun of the historical two-head cascade. The hashed word/bigram condition uses all three observable input fields, not only the note.',
         '- MiniLM has 22,713,216 frozen parameters; the head uses 3072 input features. Its 254-token windows stride192 tokens, include the tail, and aggregate mean/max embeddings plus context/question relations. No attack offsets, clean counterpart, correct task answer, labels or IDs enter the feature functions.',
         '- BIPIA commit a004b69ec0dd446e0afd461d98cb5e96e120a5d0 supplies external source records and attacks. This bounded subset uses six payloads per source and start/end insertions, not the full official BIPIA evaluation.',
         '- Final counts are 100 table source records and 50 code source records, totaling 150 clean and 900 poisoned presentations. Duplicate/near-duplicate grouping leaves84 table and 50 code source groups. Reported intervals resample source groups; payload/template reuse still limits population inference.',
         '- Source grouping precedes attack construction. Exact normalized matches and five-word-shingle Jaccard similarity at least0.90 form connected components. No group crosses training, validation and final partitions. This rule cannot detect every semantic duplicate.',
         '- Published training/test attack families overlap in Language Translation. Domain holdout is established; completely unseen attack-family evaluation is not. Encoder pretraining exposure is unknown.',
         '- A tokenizer API compatibility repair occurred before any classifier fitting or final-domain feature computation. The original code, failed logs, amendment and updated hashes are retained.','',
         '## Compute and verification','',
         f'The CPU encoder processed {result["encoder"]["unique_texts"]} unique texts in {result["encoder"]["windows"]} windows, with {result["encoder"]["encoding_seconds"]:.2f}s recorded encoding time. This is a batched local measurement, not per-request latency or T4 end-to-end overhead. Whole-process peak RSS is recorded separately and includes fitting. No GPU co-residency advantage is claimed.','',
         'The four harness tests check metadata exclusion, tail-token coverage and threshold behavior. The saved-output audit verifies 6,300 predictions and 12 domain/model summaries. Encoder parity with the published SentenceTransformer implementation is recorded in encoder_verification.json.','',
         'Conclusion: embeddings were a reasonable hypothesis to test, but this fixed small-encoder setup plus limited added diversity does not solve transfer. Preserve the negative and mixed outcomes rather than selecting a favorable domain or threshold.']
 lines += ['', '## Source-group intervals', '', '| Model / domain | Recall 95% interval | Benign rejection 95% interval |', '| --- | ---: | ---: |']
 for key,v in result['summary'].items():
  lo,hi=v['recall_ci95'];flo,fhi=v['false_positive_rate_ci95']
  lines.append(f'| {key} | {lo:.1%}–{hi:.1%} | {flo:.1%}–{fhi:.1%} |')
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
