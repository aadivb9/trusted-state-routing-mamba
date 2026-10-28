"""Self-contained reproduction of the fixed screen and repeated T4 timing."""
import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = json.loads((ROOT / 'experiments/17_external_validation/notebook_verification.json').read_text())
    names = set(base['files']) | {
        'experiments/17_external_validation/run_v2.py',
        'experiments/17_external_validation/runtime_amendment_v2.json',
        'experiments/17_external_validation/results.json',
        'src/runtime/serialization_v2.py', 'tests/test_remediation_serialization_v2.py',
        'experiments/18_runtime_validation/run_timing.py',
        'experiments/18_runtime_validation/timing_protocol.json',
        'experiments/18_runtime_validation/protocol.json', 'tests/test_remediation_timing.py'}
    payload = {name: (ROOT / name).read_text() for name in sorted(names)}
    packed = base64.b64encode(gzip.compress(json.dumps(payload).encode())).decode()
    cells = [{'cell_type': 'markdown', 'metadata': {}, 'source': [
        '# Reproduce the final competence screen and T4 timing\n',
        'The recorded experiment completed a failed clean-competence screen. This notebook is a reproduction entry point, not a new independent result. It uses the same pinned FP32 Mamba-2.8B, data, prompts, gate and runtime repair. The retained V1 failure is embedded only to validate the amendment chain. Full outputs from the recorded experiment are in the evidence package. Select a fresh T4 runtime. No prompt or checkpoint sweep is permitted. Timings use forced decisions and do not measure attack efficacy or learned-router cost.\n']}]
    def code(text):
        ast.parse(text)
        cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                      'outputs': [], 'source': text.splitlines(True)})
    code("import subprocess,sys\nsubprocess.run([sys.executable,'-m','pip','install','transformers==5.18.0','accelerate==1.15.0','huggingface-hub==1.33.0'],check=True)\nimport torch\nassert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)\nassert torch.__version__.startswith('2.11.0')\nprint(torch.cuda.get_device_name(0),torch.__version__)\n")
    code("import pathlib,base64,gzip,json\nroot=pathlib.Path('/content/mamba_final_reproduction_20261028')\nassert not root.exists(), 'Preserve previous attempts; choose a fresh runtime/directory'\nroot.mkdir()\npayload=json.loads(gzip.decompress(base64.b64decode('" + packed + "')))\nfor name,content in payload.items():\n path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)\n")
    code("subprocess.run([sys.executable,'-u','experiments/17_external_validation/run_v2.py','--execute'],cwd=root,check=True)\n")
    code("subprocess.run([sys.executable,'-u','experiments/18_runtime_validation/run_timing.py'],cwd=root,check=True)\n")
    code("import zipfile\narchive=root/'reproduction_outputs.zip'\nwith zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as out:\n for folder in ['17_external_validation','18_runtime_validation']:\n  for path in (root/'experiments'/folder).glob('*.json'):\n   out.write(path,str(path.relative_to(root)))\nfrom google.colab import files\nfiles.download(str(archive))\n")
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'accelerator': 'GPU'}, 'cells': cells}
    path = ROOT / 'notebooks/16_final_validation_reproduction.ipynb'
    path.write_text(json.dumps(notebook, indent=2) + '\n')
    tree = ast.parse(''.join(json.loads(path.read_text())['cells'][2]['source']))
    encoded = next(node.args[0].value for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == 'b64decode')
    restored = json.loads(gzip.decompress(base64.b64decode(encoded)))
    assert restored == payload
    record = {'notebook_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'files': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(names)},
              'scope': 'Serialized source equality and code syntax only; this reproduction notebook has not itself been executed.'}
    (ROOT / 'experiments/17_external_validation/final_notebook_verification.json').write_text(json.dumps(record, indent=2) + '\n')
    print('Verified', len(names), 'embedded files')


if __name__ == '__main__':
    main()
