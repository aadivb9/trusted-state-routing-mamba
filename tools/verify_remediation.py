"""Read-only checks for active revision claims and versioned new artifacts."""
import ast,base64,gzip,hashlib,json,shutil,subprocess,sys,tempfile,zipfile
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(root,name):return json.loads((root/name).read_text())
def require(value,reason):
    if not value:raise ValueError(reason)


def protocols(root):
    for folder in ('15_router_remediation','16_blinded_annotation','17_external_validation','18_runtime_validation'):
        path=root/'experiments'/folder/'protocol.json';p=json.loads(path.read_text())
        files=dict(p['files'])
        if folder=='17_external_validation':
            amendment=read(root,'experiments/17_external_validation/runtime_amendment.json')
            require(amendment['original_protocol_sha256']==sha(path),'External original protocol changed')
            require(sha(root/'experiments/17_external_validation/run.before_cuda_streaming.py')==p['files']['experiments/17_external_validation/run.py'],'Pre-amendment runner not preserved')
            files.update(amendment['files'])
        for name,h in files.items():require(sha(root/name)==h,'Frozen remediation input changed: '+name)
    r=read(root,'experiments/15_router_remediation/embedding_reconstruction.json')
    require(r['protocol_sha256']==sha(root/'experiments/15_router_remediation/protocol.json'),'Embedding protocol hash changed')
    require(r['script_sha256']==sha(root/'experiments/15_router_remediation/reconstruct_embeddings.py'),'Embedding replay code changed')
    base=root/'archive/sealed_workspace/revision_experiments/router_transfer'
    require(r['prepared_sha256']==sha(base/'prepared.json') and r['fitted_models_sha256']==sha(base/'fitted_models.json'),'Embedding replay inputs changed')
    c=read(root,'experiments/18_runtime_validation/cpu_results.json')
    require(c['status']=='passed' and c['protocol_sha256']==sha(root/'experiments/18_runtime_validation/protocol.json'),'CPU reference result stale')
    require(c['script_sha256']==sha(root/'experiments/18_runtime_validation/run.py'),'CPU reference implementation changed')
    require(c['backbone_before']==c['backbone_after'],'CPU weights changed')
    expected=read(root,'archive/sealed_workspace/revision_experiments/hispa_stress/protocol.json')['expected_backbone_sha256']
    require(c['backbone_before']==expected,'CPU replay differs from historical pinned backbone')
    require(all(x['reference_max_error']<1e-3 and x['chunk_token_max_error']<1e-3 and x['independent_storage'] and x['trusted_state_unchanged'] for x in c['checks']),'CPU parity check failed')
    notebook=root/'notebooks/15_audit_revision_external_validation.ipynb'
    record=read(root,'experiments/17_external_validation/notebook_verification.json')
    require(sha(notebook)==record['notebook_sha256'],'Prospective notebook changed')
    data=json.loads(notebook.read_text());tree=ast.parse(''.join(data['cells'][2]['source']))
    encoded=next(n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode')
    files=json.loads(gzip.decompress(base64.b64decode(encoded)))
    require(set(files)==set(record['files']),'Prospective notebook payload coverage changed')
    for name,value in files.items():require((root/name).read_bytes()==value.encode() and sha(root/name)==record['files'][name],'Prospective notebook stale: '+name)
    notebook=root/'notebooks/16_final_validation_reproduction.ipynb'
    record=read(root,'experiments/17_external_validation/final_notebook_verification.json')
    require(sha(notebook)==record['notebook_sha256'],'Final reproduction notebook changed')
    tree=ast.parse(''.join(json.loads(notebook.read_text())['cells'][2]['source']))
    encoded=next(n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode')
    files=json.loads(gzip.decompress(base64.b64decode(encoded)))
    require(set(files)==set(record['files']),'Final notebook payload coverage changed')
    for name,value in files.items():require((root/name).read_bytes()==value.encode() and sha(root/name)==record['files'][name],'Final notebook stale: '+name)


def audit_preservation(root):
    record=read(root,'audit_review/review_verification.json')
    for name,h in record['files'].items():require(sha(root/'audit_review'/name)==h,'Original audit changed: '+name)
    initial=read(root,'remediation/starting_hashes.json')['files']
    for name,h in initial.items():
        if name.startswith('audit_review/'):require(sha(root/name)==h,'Pre-remediation audit/plan changed: '+name)


def fresh_calculations(root):
    current=read(root,'experiments/15_router_remediation/results.json')
    require(current['router_sha256']==sha(root/'src/routing/guarded_router.py'),'Guarded results stale')
    require(current['code_sha256']==sha(root/'experiments/15_router_remediation/analyze.py'),'Post-hoc calculation code changed')
    with tempfile.TemporaryDirectory(prefix='remediation-recheck-') as tmp:
        dest=Path(tmp)
        names=['experiments/15_router_remediation/analyze.py','audit_review/recompute.py','audit_review/independent_checks.json',
               'src/routing/guarded_router.py','src/routing/context_router.py']
        names += ['archive/sealed_workspace/'+n for n in current['historical_input_sha256']]
        names += ['archive/sealed_workspace/external_benchmark/prepared/'+n+'.json' for n in ('clean','injected_baselines','attacks')]
        for name in names:
            p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,p)
        proc=subprocess.run([sys.executable,'experiments/15_router_remediation/analyze.py'],cwd=dest,capture_output=True,text=True,timeout=90)
        require(proc.returncode==0,proc.stderr[-2000:])
        require(read(dest,'experiments/15_router_remediation/results.json')==current,'Fresh post-audit calculation differs from stored results')


def blinding(root):
    folder=root/'experiments/16_blinded_annotation';s=read(folder,'status.json')
    require(sha(folder/'reviewer_packet.json')==s['packet_sha256'],'Reviewer packet changed')
    require(sha(folder/'private/mapping.json')==s['private_mapping_sha256'],'Reviewer mapping changed')
    private=read(folder,'private/mapping.json')
    require(s['packet_version']==2 and private['seed'].bit_length()>128,'Blinding seed is not the private V2 seed')
    require(str(private['seed']) not in (folder/'reviewer_packet.json').read_text(),'Reviewer packet leaks shuffle seed')
    amendment=read(folder,'blinding_amendment.json')
    require(amendment['status']=='withdrawn_before_review','V1 withdrawal record missing')
    for name,h in amendment['files'].items():require(sha(folder/name)==h,'V1 blinding artifact changed')
    require(s['independent_review_completed'] is True and s['human_spotcheck_completed'] is False
            and s['review_kind']=='different_model_not_human','Review kind/completion is misstated')
    require(sha(root/s['review_artifact'])==s['review_artifact_sha256'],'Returned model-review artifact changed')
    with zipfile.ZipFile(folder/'reviewer_packet.zip') as z:
        require(z.namelist()==['reviewer_packet.json'],'Reviewer ZIP leaks private material')
        require(z.read('reviewer_packet.json')==(folder/'reviewer_packet.json').read_bytes(),'Reviewer ZIP stale')


def external_preparation(root):
    base=root/'archive/sealed_workspace'
    sys.path[:0]=[str(base),str(base/'external_benchmark')]
    from external_benchmark.prepare_data import identity,format_text,read_sources,PublishedTask,ATTACKS,TASKS
    folder=root/'experiments/17_external_validation'
    prep=read(folder,'preparation.json')
    for name,h in prep['files'].items():require(sha(root/name)==h,'New external data changed: '+name)
    old=[]
    for n in ('development','clean','injected_baselines'):old+=read(base,f'external_benchmark/prepared/{n}.json')
    excluded={identity(r['text']) for r in old}
    partitions=[read(folder,'prepared/'+n+'.json') for n in ('development','clean','injected_baselines')]
    seen=set(excluded)
    source,dev=read_sources()
    lookup={r['source_id']:r for rows in [*source.values(),*dev.values()] for r in rows}
    for records in partitions:
        keys=[identity(r['text']) for r in records]
        require(len(keys)==len(set(keys)) and not set(keys)&seen,'External split overlap')
        seen.update(keys)
        for r in records:
            original=lookup[r['source_id']]
            require(r['text']==format_text(original['data']) and r['label']==original['label'],'External source/label differs')
    targets={t:[r for r in partitions[1] if r['dataset']==t] for t in TASKS}
    injected={t:[r for r in partitions[2] if r['dataset']==t] for t in TASKS}
    constructors={}
    for task in TASKS:
        obj=PublishedTask(injected[task],injected[task][0]['instruction'])
        for family,cls in ATTACKS.items():constructors[task,family]=cls(family,obj)
    attacks=read(folder,'prepared/attacks.json');require(len(attacks)==4500,'Wrong attack presentation count')
    for r in attacks:
        target,task,family,index=r['id'].split('/');index=int(index)
        original=targets[target][index]
        require(r['target_source_id']==original['source_id'] and r['injected_source_id']==injected[task][index]['source_id'],'Attack source group mismatch')
        expected=constructors[task,family].inject(original['text'],index,TASKS[target])
        require(r['untrusted_text']==expected,'Published attack constructor mismatch')
    sys.path.insert(0,str(root))
    from tools.verify_completion import external
    external(root)


def active_paper(root):
    record=read(root,'paper/build_verification.json')
    required={'paper/manuscript.md','paper/build.py','experiments/15_router_remediation/results.json',
              'experiments/15_router_remediation/embedding_reconstruction.json','experiments/18_runtime_validation/cpu_results.json',
              'experiments/17_external_validation/results_v2.json','experiments/18_runtime_validation/timing_results.json',
              'remediation/completion_20261027/model_review.json','remediation/completion_20261027/summary.json'}
    require(required<=set(record['inputs']),'Active build input coverage missing')
    for name,h in {**record['inputs'],**record['outputs']}.items():require(sha(root/name)==h,'Active paper stale: '+name)
    text=(root/'paper/manuscript.md').read_text();r=read(root,'experiments/15_router_remediation/results.json')
    require('<!-- AUDIT_REVISION_20261022 -->' in text,'Audit paper revision missing')
    for label,key,before,after in [('Two-choice ranking','paired_ranked',28,80),('Strict free generation','paired_free',20,28)]:
        value=r[key];lo,hi=value['ci95']
        row=f'| {label} | {before}/100 | {after}/100 | {value["difference"]*100:.0f} percentage points | [{lo*100:.0f}, {hi*100:.0f}] points |'
        require(text.count(row)==1,'Active paired metric mismatch: '+label)
    for key,label in [('cascade','Trained cascade'),('fast_only','Fast-only head'),('only_conflict_rule','Unfitted only-conflict'),('guarded_v1','Conservative revision')]:
        s=r['router_comparison'][key]
        row=f'| {label} | {s["malicious"]["reject"]}/100 | {s["malicious"]["abstain"]}/100 | {s["safe"]["accept"]}/100 | {s["safe"]["abstain"]}/100 |'
        require(text.count(row)==1,'Active router table mismatch: '+key)
    ml=read(root,'experiments/15_router_remediation/embedding_reconstruction.json')
    for key,m in ml['validation'].items():
        regime,rep=key.split('/');s={x['domain']:x for x in m['domains']};a,b=s['synthetic'],s['email']
        label='Synthetic' if regime=='synthetic_only' else 'Synthetic + email'
        row=f'| {label} | {rep} | {a["benign_rejected"]}/{a["benign_n"]} | {b["benign_rejected"]}/{b["benign_n"]} | {a["benign_rejected"]+b["benign_rejected"]}/110 |'
        require(text.count(row)==1,'Validation table mismatch: '+key)
    review=read(root,'remediation/narrative_review.json')
    for name,h in review['files'].items():require(sha(root/name)==h,'Active narrative changed since review: '+name)
    response=read(root,'remediation/finding_status.json')
    require([r['finding'] for r in response['findings']]==list(range(1,13)),'Findings missing from response')
    require(response['external_security_efficacy_established'] is False and response['independent_annotation_review_completed'] is True
            and response['human_annotation_review_completed'] is False,'Unsupported completion claim')
    require(response['general_defense_established'] is False and response['gpu_timing_repeated'] is False
            and response['gpu_timing_attempt_finished'] is True,'Unsupported general defense or timing claim')


def completion_results(root):
    sys.path.insert(0,str(root))
    from tools.completion_summary import make_summary,paper_block
    summary=make_summary(root)
    require(summary==read(root,'remediation/completion_20261027/summary.json'),'Completion summary differs from raw evidence')
    expected=paper_block(summary)
    require((root/'paper/completion_results.md').read_text()==expected,'Completion table view changed')
    require((root/'paper/manuscript.md').read_text().count(expected.rstrip())==1,'Completion paper claims differ from raw evidence')


CHECKS=(audit_preservation,protocols,fresh_calculations,blinding,external_preparation,active_paper,completion_results)


def verify(root):
    checks=[]
    for fn in CHECKS:
        try:fn(root);checks.append({'check':fn.__name__,'passed':True})
        except Exception as exc:checks.append({'check':fn.__name__,'passed':False,'reason':str(exc)})
    return {'status':'passed' if all(c['passed'] for c in checks) else 'failed','checks':checks}


if __name__=='__main__':
    result=verify(Path(__file__).resolve().parents[1]);print(json.dumps(result,indent=2));sys.exit(0 if result['status']=='passed' else 1)
