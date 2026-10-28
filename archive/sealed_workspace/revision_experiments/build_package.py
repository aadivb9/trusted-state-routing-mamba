"""Build the new local evidence bundle and a standalone GPU reproduction notebook."""
import ast,base64,gzip,hashlib,json,zipfile,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;OUT=ROOT/'revision_experiments'


def main():
 for folder in ['hispa_stress','competence','router_transfer']:
  assert json.loads((OUT/folder/'verification.json').read_text())['status']=='passed'
 gpu_files={}
 for folder in ['hispa_stress','competence']:
  name='revision_experiments/'+folder+'/protocol.json'
  protocol=json.loads((ROOT/name).read_text())
  for file,expected in protocol['files'].items():
   raw=(ROOT/file).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected,file
   gpu_files[file]=raw.decode()
  gpu_files[name]=(ROOT/name).read_text()
 payload=base64.b64encode(gzip.compress(json.dumps(gpu_files).encode())).decode()
 setup="import base64,gzip,json,pathlib,subprocess,sys\nroot=pathlib.Path('/content/mamba_revision_reproduction')\nassert not root.exists(), 'Use a fresh directory; preserve prior attempts'\nroot.mkdir()\nfiles=json.loads(gzip.decompress(base64.b64decode('"+payload+"')))\nfor name,value in files.items():\n p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)\nimport torch,transformers\nassert torch.cuda.is_available()\nprint(torch.cuda.get_device_name(0),torch.__version__,transformers.__version__)\n"
 cells=[dict(cell_type='markdown',metadata={},source=['# Reproduce the two GPU diagnostics\n','These are computational repeats on previously observed data. The recorded runs used FP32, a T4, torch2.11.0+cu130 and transformers5.18.0. Use a fresh runtime/directory, preserve all outcomes, and do not count repetitions as new independent samples. Large model weights download from pinned public revisions. No credentials are embedded.\n']),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=setup.splitlines(True))]
 for folder in ['hispa_stress','competence']:
  code=f"subprocess.run([sys.executable,'-u','revision_experiments/{folder}/run.py'],cwd=root,check=True)\n"
  cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(True)))
 code="import zipfile\narchive=root/'gpu_results.zip'\nwith zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:\n for folder in ['hispa_stress','competence']:\n  p=root/'revision_experiments'/folder/'results.json';z.write(p,str(p.relative_to(root)))\nfrom google.colab import files\nfiles.download(str(archive))\n"
 cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(True)))
 notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),accelerator='GPU'),cells=cells)
 notebook_path=OUT/'gpu_reproduction.ipynb';notebook_path.write_text(json.dumps(notebook,indent=2)+'\n')
 # Verify actual serialized notebook payload against every source byte.
 notebook=json.loads(notebook_path.read_text());tree=ast.parse(''.join(notebook['cells'][1]['source']))
 encoded=next(n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode')
 decoded=json.loads(gzip.decompress(base64.b64decode(encoded)))
 for name,text in decoded.items():assert (ROOT/name).read_bytes()==text.encode(),name
 (OUT/'notebook_verification.json').write_text(json.dumps(dict(status='passed',embedded_files=len(decoded),notebook_sha256=hashlib.sha256(notebook_path.read_bytes()).hexdigest(),scope='Payload parity only; no extra GPU replication'),indent=2)+'\n')
 code_names=['hispa_stress/run.py','hispa_stress/audit.py','competence/run.py','competence/audit.py','router_transfer/model.py','router_transfer/prepare.py','router_transfer/run.py','router_transfer/audit.py','router_transfer/test_controls.py','router_transfer/verify_encoder.py']
 parts=['COMPLETED FOLLOW-UP CODE\n\n'+(OUT/'README.md').read_text()]
 for name in code_names+['hispa_stress/protocol.json','competence/protocol.json','router_transfer/protocol.json','router_transfer/runtime_amendment.json']:
  parts.extend(['\n'+'='*76+'\nrevision_experiments/'+name+'\n'+'='*76+'\n',(OUT/name).read_text()])
 (OUT/'code_review.txt').write_text('\n'.join(parts))
 names=set()
 for folder in [OUT,ROOT/'paper']:
  for p in folder.rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts and not p.name.startswith('.') and p.name not in ['package_verification.json','evidence_manifest.json'] and p.suffix!='.pyc':names.add(str(p.relative_to(ROOT)))
 names.update(gpu_files)
 names.update(['adaptive_memory_router.py','adaptive_router_experiment/dataset.json','external_benchmark/audit_development.py'])
 # Include audit dependencies and paper build inputs, not just inference inputs.
 names.add('final_hispa_audit/results.json')
 names.update(json.loads((ROOT/'paper/build_verification.json').read_text())['inputs'])
 names.update(['README.md','FINAL_RESEARCH_SUMMARY.md','PAPER_INTRODUCTION.md','research_closeout/STATUS.md'])
 names.update(['.forge/README.md','.forge/check.py','.forge/sync_status.py','.forge/test_check.py','.forge/narrative_review.json'])
 manifest={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sorted(names)}
 manifest_path=OUT/'evidence_manifest.json';manifest_path.write_text(json.dumps(dict(scope='Local revision evidence with third-party provenance; no model weights or public release',files=manifest),indent=2)+'\n')
 archive=ROOT/'revision_experiments_evidence.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for name in sorted(names):z.write(ROOT/name,name)
  z.write(manifest_path,str(manifest_path.relative_to(ROOT)))
 with zipfile.ZipFile(archive) as z:
  for name,expected in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==expected,name
 # A byte-perfect archive can still omit a runtime dependency. Exercise the
 # documented saved-output audits in an empty extraction, without the workspace.
 with tempfile.TemporaryDirectory(prefix='revision-package-check-') as tmp:
  with zipfile.ZipFile(archive) as z:z.extractall(tmp)
  for folder in ['hispa_stress','competence','router_transfer']:
   proc=subprocess.run([sys.executable,f'revision_experiments/{folder}/audit.py'],cwd=tmp,capture_output=True,text=True,timeout=90)
   if proc.returncode:raise RuntimeError(f'Isolated {folder} audit failed: {proc.stderr}')
   assert json.loads((Path(tmp)/'revision_experiments'/folder/'verification.json').read_text())==json.loads((OUT/folder/'verification.json').read_text()),folder
 result=dict(status='passed',files_verified=len(manifest),isolated_audits_passed=3,archive=archive.name,archive_bytes=archive.stat().st_size,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
 (OUT/'package_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
