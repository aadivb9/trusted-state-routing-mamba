"""Fit the small evidence classifiers on CPU; select only on validation."""
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import adaptive_memory_router as routing
from dual_state_mamba import binary_metrics, table

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'adaptive_router_experiment'


def inputs(record):
    return record['trusted_context'],record['span'],record['spec']['stem']


def fit(train,validation,detailed):
    x=[routing.features(*inputs(r),detailed=detailed)[0] for r in train]
    y=[int(r['attack_label']=='malicious') for r in train]
    vx=[routing.features(*inputs(r),detailed=detailed)[0] for r in validation]
    vy=[int(r['attack_label']=='malicious') for r in validation]
    head={'weights':[0.]*len(routing.FEATURE_NAMES),'bias':0.}
    rng=random.Random(2709)
    best=None
    best_loss=float('inf')
    for epoch in range(100):
        order=list(range(len(train)));rng.shuffle(order)
        rate=.15/math.sqrt(1+epoch/10)
        for i in order:
            error=routing.probability(head,x[i])-y[i]
            head['bias']-=rate*error
            for j,v in enumerate(x[i]):head['weights'][j]-=rate*(error*v+.001*head['weights'][j])
        loss=-sum(label*math.log(max(p,1e-9))+(1-label)*math.log(max(1-p,1e-9)) for vec,label in zip(vx,vy) for p in [routing.probability(head,vec)])/len(vy)
        if loss<best_loss:
            best_loss=loss;best=copy.deepcopy(head);best.update(epoch=epoch+1,validation_loss=loss)
    return best


def train():
    if (OUT/'router.json').exists():raise RuntimeError('Trained router already exists; preserve the sealed run.')
    dataset=OUT/'dataset.json'
    records=json.loads(dataset.read_text())['records']
    train=[r for r in records if r['split']=='train'];validation=[r for r in records if r['split']=='validation']
    started=time.perf_counter()
    router={'type':'adaptive_context_evidence_logistic','feature_names':routing.FEATURE_NAMES,
            'fast':fit(train,validation,False),'detailed':fit(train,validation,True),
            'threshold':.5,'uncertainty_band':[.15,.85],'backbone_trainable_parameters':0,
            'router_trainable_parameters':2*(len(routing.FEATURE_NAMES)+1),
            'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest()}
    val=[routing.route(router,*inputs(r)) for r in validation]
    labels=[r['attack_label']=='malicious' for r in validation]
    router['threshold']=max([i/100 for i in range(10,91)],key=lambda t:(
        binary_metrics(labels,[v['risk']>=t for v in val])['accuracy'], -abs(t-.5)))
    router['training_seconds']=time.perf_counter()-started
    (OUT/'router.json').write_text(json.dumps(router,indent=2)+'\n')
    print('Trained two evidence classifiers:',router['router_trainable_parameters'],'parameters;',round(router['training_seconds'],2),'seconds')


if __name__=='__main__':train()
