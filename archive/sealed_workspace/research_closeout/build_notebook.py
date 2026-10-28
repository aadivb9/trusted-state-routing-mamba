"""Seal and package the final diagnostic before GPU execution."""
import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
NAMES=['dual_state_mamba.py','adaptive_memory_router.py','adaptive_router_experiment/router.json',
       'research_closeout/PROTOCOL.md','research_closeout/run_external_final.py',
       'research_closeout/router_coverage.py','external_benchmark/run_competence.py',
       'external_benchmark/upstream/OpenPromptInjection/evaluator/utils.py',
       'external_benchmark/prepared/clean.json','external_benchmark/prepared/injected_baselines.json',
       'external_benchmark/prepared/attacks.json']


def main():
    files={name:(ROOT/name).read_text() for name in NAMES}
    seal=dict(stage='post-development project close-out', final_attack_generation=False,
              model_revision='96c48e0292b63f5346b6d30061af2551f7101e26',
              files={n:hashlib.sha256(c.encode()).hexdigest() for n,c in files.items()})
    seal_path=ROOT/'research_closeout/seal.json'
    serialized=json.dumps(seal,indent=2)+'\n'
    if seal_path.exists():
        assert seal_path.read_text()==serialized, 'Do not overwrite an existing seal'
    else:
        seal_path.write_text(serialized)
    files['research_closeout/seal.json']=serialized
    payload=base64.b64encode(gzip.compress(json.dumps(files).encode(),mtime=0)).decode()
    setup=f'''import base64, gzip, json, os, pathlib, subprocess, sys
root=pathlib.Path('/content/mamba_final_closeout_20261011')
root.mkdir(exist_ok=True)
files=json.loads(gzip.decompress(base64.b64decode({payload!r})))
for name,content in files.items():
    path=root/name
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): assert path.read_text()==content
    path.write_text(content)
print('Prepared final diagnostic with sealed inputs')
'''
    run='''result_dir=root/'run_01'
assert not result_dir.exists()
log=open(root/'run_01.log','w')
closeout_process=subprocess.Popen([sys.executable,'-u',str(root/'research_closeout/run_external_final.py'),'--output',str(result_dir)],cwd=root,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'})
print('Launched',closeout_process.pid)
'''
    status='''print('EXIT',closeout_process.poll())
print((root/'run_01.log').read_text()[-2500:])
result_path=root/'run_01/results.json'
if result_path.exists():
    raw=result_path.read_bytes()
    data=json.loads(raw)
    print('STATUS',data['status'],'ROWS',len(data['rows']))
    print('RESULT_GZIP_B64',base64.b64encode(gzip.compress(raw)).decode())
'''
    cells=[dict(cell_type='markdown',metadata={},source=['# Final diagnostic extension\n','600 baseline generations and a separate 4,500-attack router diagnostic. This is not a completed attack-efficacy evaluation. Re-run the status cell to export checkpoints.\n'])]
    for code in [setup,run,status]:
        ast.parse(code)
        cells.append(dict(cell_type='code',metadata={},source=code.splitlines(keepends=True),outputs=[],execution_count=None))
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={'accelerator':'GPU','kernelspec':{'name':'python3','display_name':'Python 3'}},cells=cells)
    (ROOT/'research_closeout/final_diagnostic.ipynb').write_text(json.dumps(notebook,indent=1)+'\n')
    print('Sealed',len(NAMES),'files and built notebook')


if __name__=='__main__':
    main()
