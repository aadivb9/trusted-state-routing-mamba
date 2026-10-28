"""Fit fixed heads, freeze them, then evaluate untouched table/code domains."""
import hashlib,json,os,platform,resource,sys,time,warnings
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.exceptions import ConvergenceWarning
from model import Encoder,lexical,symbolic


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,d):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)


def operating_threshold(y,p):
    candidates=np.unique(np.r_[p,np.nextafter(p.max(),np.inf)])
    eligible=[]
    for t in candidates:
        reject=p>=t;fpr=reject[y==0].mean();tpr=reject[y==1].mean()
        if fpr<=.05:eligible.append((tpr,-fpr,t))
    best=max(eligible)
    return float(best[2]),dict(tpr=float(best[0]),fpr=float(-best[1]),n=len(y))


def metrics(rows,p,t):
    y=np.array([r['label'] for r in rows]);pred=p>=t
    d=dict(n=len(rows),attack_n=int(y.sum()),benign_n=int((y==0).sum()),
           true_positive=int(((y==1)&pred).sum()),false_positive=int(((y==0)&pred).sum()),
           recall=float(pred[y==1].mean()),false_positive_rate=float(pred[y==0].mean()),
           balanced_accuracy=float((pred[y==1].mean()+(~pred[y==0]).mean())/2),
           roc_auc=float(roc_auc_score(y,p)),average_precision=float(average_precision_score(y,p)),
           brier=float(np.mean((p-y)**2)))
    groups=sorted({r['source_group'] for r in rows})
    totals=[]
    for group in groups:
        mask=np.array([r['source_group']==group for r in rows]);a=mask&(y==1);b=mask&(y==0)
        totals.append([pred[a].sum(),a.sum(),pred[b].sum(),b.sum()])
    a=np.array(totals);rng=np.random.default_rng(20261016);draw=a[rng.integers(0,len(a),size=(2000,len(a)))].sum(1)
    d.update(source_groups=len(groups),recall_ci95=np.quantile(draw[:,0]/draw[:,1],[.025,.975]).tolist(),
             false_positive_rate_ci95=np.quantile(draw[:,2]/draw[:,3],[.025,.975]).tolist())
    return d


def main():
    assert not (OUT/'fitted_models.json').exists(),'Do not refit observed final domains'
    protocol=json.loads((OUT/'protocol.json').read_text())
    amendment=json.loads((OUT/'runtime_amendment.json').read_text())
    for name,h in protocol['files'].items():
        if name in amendment['files']:
            assert sha(ROOT/(name+'.before_tokenizer_fix'))==h,name
            assert sha(ROOT/name)==amendment['files'][name],name
        else:assert sha(ROOT/name)==h,name
    all_rows=json.loads((OUT/'prepared.json').read_text())
    development=[r for r in all_rows if r['split'] in ['train','validation']]
    final=[r for r in all_rows if r['split']=='final']
    encoder=Encoder()
    start=time.perf_counter()
    features={'symbolic':symbolic(development),'bigrams':lexical(development),'embeddings':encoder.transform(development)}
    train_times={};models={}
    labels=np.array([r['label'] for r in development]);val=np.array([r['split']=='validation' for r in development])
    warnings.simplefilter('error',ConvergenceWarning)
    for regime in ['synthetic_only','synthetic_plus_email']:
        train=np.array([r['split']=='train' and (regime=='synthetic_plus_email' or r['domain']=='synthetic') for r in development])
        for representation,x in features.items():
            begin=time.perf_counter();scaler=None
            if representation!='bigrams':
                scaler=StandardScaler().fit(x[train]);scaled=scaler.transform(x)
            else:scaled=x
            classifier=LogisticRegression(C=1.,max_iter=2000,class_weight='balanced',solver='lbfgs',random_state=20261016).fit(scaled[train],labels[train])
            probs=classifier.predict_proba(scaled[val])[:,1];threshold,val_metrics=operating_threshold(labels[val],probs)
            key=regime+'/'+representation
            models[key]=dict(weights=classifier.coef_[0].tolist(),intercept=float(classifier.intercept_[0]),
                             mean=scaler.mean_.tolist() if scaler else None,scale=scaler.scale_.tolist() if scaler else None,
                             threshold=threshold,validation=val_metrics,train_n=int(train.sum()),features=x.shape[1],
                             fit_seconds=time.perf_counter()-begin)
            print('FITTED',key,'validation',val_metrics,flush=True)
    write('fitted_models.json',models)
    write('fitted_seal.json',dict(protocol_sha256=sha(OUT/'protocol.json'),models_sha256=sha(OUT/'fitted_models.json'),runtime_amendment_sha256=sha(OUT/'runtime_amendment.json'),
                                rule='Written before final-domain featurization or prediction; no subsequent refitting',
                                final_input_sha256=sha(OUT/'prepared.json')))
    # Everything below uses frozen models. No fit, scaling updates, or threshold selection.
    final_features={'symbolic':symbolic(final),'bigrams':lexical(final),'embeddings':encoder.transform(final)}
    rows=[];summary={}
    for key,m in models.items():
        rep=key.split('/')[1];x=final_features[rep]
        if m['mean'] is not None:x=(x-np.array(m['mean']))/np.array(m['scale'])
        begin=time.perf_counter();logits=np.asarray(x@np.array(m['weights'])+m['intercept']).reshape(-1)
        p=1/(1+np.exp(-np.clip(logits,-700,700)));head_seconds=time.perf_counter()-begin
        for domain in ['table','code']:
            mask=np.array([r['domain']==domain for r in final]);subset=[r for r in final if r['domain']==domain]
            summary[key+'/'+domain]=metrics(subset,p[mask],m['threshold'])
        for r,prob in zip(final,p):rows.append(dict(model=key,id=r['id'],domain=r['domain'],source_group=r['source_group'],label=r['label'],risk=float(prob),rejected=bool(prob>=m['threshold']),position=r['position'],attack_family=r['attack_family']))
        print('EVALUATED',key,flush=True)
    result=dict(status='complete',evaluation='frozen_router_domain_transfer_detection_only',rows=rows,summary=summary,
                protocol_sha256=sha(OUT/'protocol.json'),runtime_amendment_sha256=sha(OUT/'runtime_amendment.json'),fitted_models_sha256=sha(OUT/'fitted_models.json'),
                encoder=encoder.stats,elapsed_seconds=time.perf_counter()-start,platform=platform.platform(),
                process_max_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                memory_unit='bytes on macOS; whole process peak including fitting and evaluation',
                scope='Detection only. No Mamba task utility, attack success, rollback performance, or official BIPIA reproduction claim.')
    write('results.json',result);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
