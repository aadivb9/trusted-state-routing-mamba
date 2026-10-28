"""Build an offline-input Colab notebook for the sealed generalization pilot."""
import base64
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build():
    from evaluate_generalization import verify
    protocol = verify()
    seal = json.loads((ROOT / 'generalization_experiment/seal.json').read_text())
    names = sorted(set(protocol['frozen_files']) | set(seal['files']) | {
        'generalization_experiment/seal.json', 'write_generalization_report.py',
    })
    amended = (ROOT / 'generalization_experiment/runtime_amendment.json').exists()
    if amended:
        names += ['evaluate_generalization_boundary_fix.py', 'generalization_experiment/runtime_amendment.json']
        amendment = json.loads((ROOT / 'generalization_experiment/runtime_amendment.json').read_text())
        for name, expected in amendment['files'].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
        names += [str(path.relative_to(ROOT)) for path in
                  sorted((ROOT / 'generalization_experiment/interrupted_run').glob('*')) if path.is_file()]
    files = {name: base64.b64encode((ROOT / name).read_bytes()).decode() for name in names}
    payload = base64.b64encode(gzip.compress(json.dumps(files).encode(), mtime=0)).decode()
    cells = []

    def add(kind, source):
        cell = {'cell_type': kind, 'metadata': {}, 'id': f'generalization-{len(cells):02d}',
                'source': source.splitlines(keepends=True)}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add('markdown', '''# Frozen-router generalization on research abstracts

Run top to bottom in a **Colab T4 GPU** runtime. This 32-paper pilot compares original Mamba-130M with the same frozen backbone and existing adaptive router across clean, safe-addition, early-attack, late-attack and mixed-note conditions.

The original inputs and evaluator were sealed before outcomes. A separately hashed runtime amendment repairs token boundaries after the first GPU attempt stopped; the original files and interrupted evidence are preserved. No tasks, labels, router parameters, prompts, or success criteria changed. No router retraining, API calls, or automatic package installation occurs. PyTorch, Transformers, NumPy and huggingface_hub must already be available. The only model download is `state-spaces/mamba-130m-hf`.

Candidate ranking and free generation are separate metrics. Failure of the clean-context or attack gate is a valid stopped outcome; the experiment must not retune itself to pass.
''')
    add('code', 'import base64, datetime, gzip, hashlib, json, shutil, subprocess, sys\nfrom pathlib import Path\n'
        'work=Path("generalization_mamba_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))\nwork.mkdir()\n'
        f'files=json.loads(gzip.decompress(base64.b64decode({payload!r})))\n'
        'for name,encoded in files.items():\n    destination=work/name\n    destination.parent.mkdir(parents=True,exist_ok=True)\n    destination.write_bytes(base64.b64decode(encoded))\n'
        'for document,field in [("protocol.json","frozen_files"),("seal.json","files")]:\n'
        '    manifest=json.loads((work/"generalization_experiment"/document).read_text())\n'
        '    for name,expected in manifest[field].items():\n        assert hashlib.sha256((work/name).read_bytes()).hexdigest()==expected,name\n'
        'print("Verified frozen inputs:",work)\n')
    add('markdown', '## Check the runtime\n\nStop before downloading a model if CUDA is unavailable.\n')
    add('code', 'import torch, transformers, numpy, huggingface_hub\n'
        'assert torch.cuda.is_available(), "Select a Colab T4 GPU runtime."\n'
        'print(torch.cuda.get_device_name(0), "torch", torch.__version__, "transformers", transformers.__version__)\n')
    add('markdown', '''## Evaluate

First measure clean and empty-context baselines. Structural cache probes compare forced accept/reject decisions with filtering independently of answer accuracy. If the baseline gate passes, test the original model on incoming notes; only an observed late-attack accuracy drop permits the defense comparison.

The full run also checks filtering equivalence under actual routing decisions and synchronized generation speed. Routing always uses the immutable trusted abstract; accepted notes are not promoted into the router's reference text.
''')
    entrypoint = 'evaluate_generalization_boundary_fix.py' if amended else 'evaluate_generalization.py'
    add('code', f'subprocess.run([sys.executable,"-u",{entrypoint!r}],cwd=work,check=True)\n')
    add('markdown', '## Read the outcome and save artifacts\n\nThis cell supports both completed and gate-stopped experiments.\n')
    add('code', 'subprocess.run([sys.executable,"write_generalization_report.py"],cwd=work,check=True)\n'
        'from IPython.display import Markdown, display, FileLink\n'
        'display(Markdown((work/"generalization_experiment/REPORT.md").read_text()))\n'
        'archive=shutil.make_archive(str(work),"zip",root_dir=work)\ndisplay(FileLink(archive))\n')
    result_path = ROOT / 'generalization_experiment/results.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        recorded = ['## Recorded GPU outcome', '', f"Status: **{result['status']}**.", '',
                    f"Clean ranked accuracy: {result['clean_gate']['clean']['correct']}/32; "
                    f"empty-context ranked accuracy: {result['clean_gate']['empty_context']['correct']}/32."]
        if 'summary' in result:
            recorded += ['', '| Condition | Original ranked | Dual ranked |', '| --- | ---: | ---: |']
            for condition in ['clean', 'safe_addition', 'attack_early', 'attack_late', 'mixed_note']:
                values = [result['summary'][v][condition]['candidate']['accuracy'] for v in ['original', 'dual_state']]
                recorded.append(f'| {condition} | {values[0]:.1%} | {values[1]:.1%} |')
            recorded += ['', f"Full-path dual/original throughput ratio: {result['speed_ratio']:.3f}."]
        recorded += ['', 'These are saved measurements. Rerunning this notebook repeats the same pilot; it is not a new independent test.']
        add('markdown', '\n'.join(recorded) + '\n')
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'cells': cells, 'metadata': {
        'accelerator': 'GPU', 'kernelspec': {'name': 'python3', 'language': 'python', 'display_name': 'Python 3'},
        'colab': {'name': 'generalization_mamba.ipynb', 'provenance': []},
    }}
    (ROOT / 'generalization_mamba.ipynb').write_text(json.dumps(notebook, indent=1) + '\n')
    print('Built generalization_mamba.ipynb with', len(files), 'embedded files.')


if __name__ == '__main__':
    build()
