"""Three pinned encoders, document/span heads, calibration-only selection."""
import collections,gc,hashlib,importlib.metadata,json,math,sys,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from src.routing.embedding_followup import Encoder,MODELS
from src.validation.calibration import select_threshold,binomial_upper

def save(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def aggregate(records,risks,granularity):
 return [dict(domain=r['domain'],source_group=r['source_group'],label=r['label'],risk=float(max(v))) for r,v in zip(records,risks)]
def threshold_metrics(rows,t):
 grouped=collections.defaultdict(lambda:collections.defaultdict(list))
 for r in rows:grouped[r['domain']][r['source_group'],r['label']].append(r['risk']>=t)
 result={}
 for d,g in grouped.items():
  clean=[any(v) for (s,l),v in g.items() if l==0];bad=[sum(v)/len(v) for (s,l),v in g.items() if l==1];k=sum(clean)
  result[d]=dict(benign_source_groups=len(clean),benign_groups_rejected=k,fpr=k/len(clean),source_balanced_recall=sum(bad)/len(bad))
 return result
def operating_point(rows):
 best=None
 for t in sorted({r['risk'] for r in rows}|{math.nextafter(max(r['risk'] for r in rows),math.inf)}):
  m=threshold_metrics(rows,t)
  if any(r['fpr']>.05 for r in m.values()):continue
  quality=(min(r['source_balanced_recall'] for r in m.values()),sum(r['source_balanced_recall'] for r in m.values())/len(m),-max(r['fpr'] for r in m.values()),t)
  if best is None or quality>best[0]:best=(quality,t,m)
 assert best is not None
 for m in best[2].values():m['fpr_upper']=binomial_upper(m['benign_groups_rejected'],m['benign_source_groups'],.05/3)
 return dict(threshold=best[1],domains=best[2],selection_quality=list(best[0][:3]),scope='Exploratory point-FPR <=5% operating point. This is not a confidence-qualified safety operating point.')
def predict(e,head,record):
 x=e.features(record,head['granularity']);mean=np.asarray(head['mean']);scale=np.asarray(head['scale']);coef=np.asarray(head['coef'])
 logits=((x-mean)/scale)@coef+head['intercept'];return (1/(1+np.exp(-np.clip(logits,-40,40)))).tolist()
def evaluate(records,risks,t):
 from sklearn.metrics import roc_auc_score,average_precision_score
 rows=aggregate(records,risks,'');m=threshold_metrics(rows,t);scores=[r['risk'] for r in rows];labels=[r['label'] for r in rows]
 return dict(domains=m,auroc=float(roc_auc_score(labels,scores)),average_precision=float(average_precision_score(labels,scores)),presentations=len(rows))

def main():
 import torch
 from sklearn.linear_model import LogisticRegression
 from sklearn.preprocessing import StandardScaler
 p=json.loads((OUT/'protocol.json').read_text())
 for name,h in p['files'].items():assert sha(ROOT/name)==h,'Frozen input changed: '+name
 assert not (OUT/'results.json').exists(),'Preserve prior run'
 data=json.loads((OUT/'prepared.json').read_text());train=[r for r in data if r['split']=='train'];cal=[r for r in data if r['split']=='calibration'];final=[r for r in data if r['split']=='final']
 result=dict(status='running',protocol_sha256=sha(OUT/'protocol.json'),versions={n:importlib.metadata.version(n) for n in ['torch','transformers','scikit-learn','numpy','tokenizers','huggingface-hub']},heads={},final={},events=[])
 path=OUT/'results.json';save(path,result);heads={}
 try:
  for name in MODELS:
   e=Encoder(name,'mps');e.populate_records(train+cal)
   for granularity in ['document','span']:
    xs=[];ys=[];groups=[];seen=set()
    for r in train:
     values=e.features(r,granularity);labels=[r['label']] if granularity=='document' else [s['label'] for s in r['spans']]
     for value,label,span in zip(values,labels,[r['note']] if granularity=='document' else [s['text'] for s in r['spans']]):
      identity=(r['source_group'],r['trusted'],r['question'],span,label)
      if identity in seen:continue
      seen.add(identity);xs.append(value);ys.append(label);groups.append((r['source_group'],label))
    x=np.stack(xs);y=np.asarray(ys);counts=collections.Counter(groups);weights=np.asarray([1/counts[g] for g in groups]);weights*=len(weights)/weights.sum()
    scaler=StandardScaler().fit(x,sample_weight=weights);fit=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs',max_iter=2000,random_state=20261029).fit(scaler.transform(x),y,sample_weight=weights)
    if int(max(fit.n_iter_))>=2000:raise RuntimeError('Classifier did not converge; no silent solver change')
    head=dict(encoder=name,granularity=granularity,mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),coef=fit.coef_[0].tolist(),intercept=float(fit.intercept_[0]),train_examples=len(y),train_source_groups=len(set(g[0] for g in groups)),iterations=int(max(fit.n_iter_)))
    risks=[predict(e,head,r) for r in cal];rows=aggregate(cal,risks,granularity);point=operating_point(rows)
    confidence=select_threshold(rows,['email','table','code'])
    head['threshold']=point['threshold'];key=name+'/'+granularity;heads[key]=head
    result['heads'][key]=dict(encoder=e.stats.copy(),training={k:head[k] for k in ['train_examples','train_source_groups','iterations']},point_operating=point,confidence_calibration=confidence)
    save(path,result);print('FITTED',key,point['selection_quality'],confidence['status'],flush=True)
    del x,y,fit,xs;gc.collect()
   del e;gc.collect();torch.mps.empty_cache()
  # Select only on known-domain calibration. No final embedding features read yet.
  keys=list(heads);chosen=max(keys,key=lambda k:(*result['heads'][k]['point_operating']['selection_quality'],-keys.index(k)))
  save(OUT/'frozen_heads.json',heads);selection=dict(head=chosen,selection_basis='Calibration only; no final-domain scores',confidence_status=result['heads'][chosen]['confidence_calibration']['status'],heads_sha256=sha(OUT/'frozen_heads.json'),timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
  save(OUT/'selection.json',selection);result['selection']=selection;result['events'].append('All six heads and calibration-only selection frozen before final encoder features');save(path,result)
  for name in MODELS:
   e=Encoder(name,'mps');e.populate_records(final)
   for granularity in ['document','span']:
    key=name+'/'+granularity;head=heads[key];risks=[predict(e,head,r) for r in final];evaluation=evaluate(final,risks,head['threshold'])
    predictions=[]
    for r,scores in zip(final,risks):
     mask=[s>=head['threshold'] for s in scores]
     if granularity=='document':mask*=len(r['spans'])
     bad=[s['label']==1 for s in r['spans']];predictions.append(dict(id=r['id'],source_group=r['source_group'],label=r['label'],scores=scores,rejected_spans=mask,payload_spans_removed=sum(a and b for a,b in zip(mask,bad)),payload_spans=sum(bad),benign_characters=sum(s['right']-s['left'] for s in r['spans'] if not s['label']),benign_characters_removed=sum(s['right']-s['left'] for s,b in zip(r['spans'],mask) if b and not s['label'])))
    result['final'][key]=dict(metrics=evaluation,predictions=predictions);save(path,result);print('FINAL',key,evaluation,flush=True)
   del e;gc.collect();torch.mps.empty_cache()
  result['status']='complete'
 except Exception as e:result.update(status='failed',error=str(e),traceback=traceback.format_exc());save(path,result);raise
 save(path,result);print('FOLLOWUP_STATUS',result['status'],flush=True)
if __name__=='__main__':main()
