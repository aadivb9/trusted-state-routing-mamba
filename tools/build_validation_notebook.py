"""Embed the frozen prospective external study for a fresh Colab runtime.

Payload verification is not a GPU execution or validation of the CUDA runner.
"""
import ast,base64,gzip,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'experiments/17_external_validation'
    p=json.loads((folder/'protocol.json').read_text());a=json.loads((folder/'runtime_amendment.json').read_text())
    files=set(p['files'])|set(a['files'])
    files.update(['experiments/17_external_validation/protocol.json','experiments/17_external_validation/runtime_amendment.json',
                  'src/routing/context_router.py'])
    # Transitive imports of the retained token-boundary adapter.
    files.update('archive/sealed_workspace/'+n for n in ['evaluate_generalization.py','adaptive_memory_router.py','cloze_dual_state_eval.py','clean_baseline_diagnostic.py'])
    payload={n:(ROOT/n).read_text() for n in sorted(files)}
    encoded=base64.b64encode(gzip.compress(json.dumps(payload).encode())).decode()
    cells=[{'cell_type':'markdown','metadata':{},'source':['# Prospective external validation — NOT EXECUTED\n',
            'This notebook has only been payload-checked locally. It is not a completed benchmark. Use a fresh CUDA runtime with PyTorch 2.11.0, and preserve all outcomes. The fixed 72-example gate must pass on all three tasks before attack-conditioned generation. No prompt or checkpoint sweep is permitted. The model is FP32 Mamba-2.8B, and a passing gate would lead to a long multi-arm evaluation.\n']}]
    def cell(code):cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)})
    cell("import sys,subprocess\nsubprocess.run([sys.executable,'-m','pip','install','transformers==5.18.0','accelerate==1.15.0','huggingface-hub==1.33.0'],check=True)\nimport torch\nassert torch.cuda.is_available(), 'CUDA required; no checkpoint download attempted'\nassert torch.__version__.startswith('2.11.0'), 'Use the pinned torch 2.11.0 CUDA runtime; do not silently substitute versions'\nprint(torch.cuda.get_device_name(0),torch.__version__)\n")
    cell("import base64,gzip,json,pathlib\nroot=pathlib.Path('/content/mamba_audit_revision_20261022')\nassert not root.exists(), 'Preserve previous attempts; use a fresh runtime/directory'\nroot.mkdir()\nfiles=json.loads(gzip.decompress(base64.b64decode('"+encoded+"')))\nfor name,value in files.items():\n p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)\n")
    cell("subprocess.run([sys.executable,'-u','experiments/17_external_validation/run.py','--execute'],cwd=root,check=True)\n")
    cell("import zipfile\narchive=root/'external_validation_results.zip'\nwith zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:\n for p in (root/'experiments/17_external_validation').glob('*.json'):\n  z.write(p,str(p.relative_to(root)))\nfrom google.colab import files\nfiles.download(str(archive))\n")
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'accelerator':'GPU'},'cells':cells}
    path=ROOT/'notebooks/15_audit_revision_external_validation.ipynb'
    path.write_text(json.dumps(notebook,indent=2)+'\n')
    actual=json.loads(path.read_text());tree=ast.parse(''.join(actual['cells'][2]['source']))
    packed=next(n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode')
    restored=json.loads(gzip.decompress(base64.b64decode(packed)))
    for n,value in restored.items():assert (ROOT/n).read_bytes()==value.encode(),n
    record={'status':'passed','embedded_files':len(restored),'notebook_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'files':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in restored},
            'scope':'Serialized payload equality only; no new GPU model execution or validation of the prospective CUDA runner.'}
    (folder/'notebook_verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='files'},indent=2))


if __name__=='__main__':main()
