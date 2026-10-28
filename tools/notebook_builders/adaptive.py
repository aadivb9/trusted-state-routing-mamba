"""Package the adaptive experiment for one top-to-bottom Colab execution."""
import base64
import gzip
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
FILES=[
    'dual_state_mamba.py','cloze_dual_state_eval.py','clean_baseline_diagnostic.py',
    'adaptive_memory_router.py','train_adaptive_router.py','generate_router_training_data.py',
    'generate_adaptive_benchmark.py','generate_text_router_benchmark.py',
    'audit_adaptive_benchmark.py','audit_text_router_benchmark.py',
    'evaluate_adaptive_mamba.py','evaluate_text_router_benchmark.py','build_text_router_notebook.py',
    'text_router_benchmark_archive.zip','write_adaptive_report.py',
    'text_router_benchmark/dataset.json','text_router_benchmark/answers.json','text_router_benchmark/results.json',
    'adaptive_router_experiment/dataset.json','adaptive_router_experiment/router.json',
    'adaptive_router_experiment/protocol.json','adaptive_router_experiment/evaluation_seal.json',
    'adaptive_router_experiment/independent_benchmark/dataset.json',
    'adaptive_router_experiment/independent_benchmark/semantic_audit.json',
]


def payload():
    files={name:base64.b64encode((ROOT/name).read_bytes()).decode() for name in FILES}
    return base64.b64encode(gzip.compress(json.dumps(files).encode(),mtime=0)).decode()


def build():
    cells=[]
    def add(kind,text):
        cell={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True),'id':f'adaptive-{len(cells):02d}'}
        if kind=='code':cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    add('markdown','''# Adaptive memory routing for Mamba-130M

Run top to bottom with a **Colab T4 GPU**. This compares original Mamba with the same frozen backbone and a retrained, context-aware dual-memory router. Two small classifiers have 30 learned parameters. Detailed evidence checking is used for ambiguous notes. This is a restricted fact-verification pipeline, not general reasoning or a trained internal dual-SSM architecture.

No API calls or package installation occur here. Only `state-spaces/mamba-130m-hf` is downloaded. PyTorch and Transformers must already be available. The notebook contains the training data, trained router, independent evaluation notes, historical regression inputs, and frozen implementation.
''')
    add('markdown','## Phase 0 — Restore and verify the experiment\n')
    add('code','import base64,gzip,json,datetime,subprocess,sys,hashlib,shutil\nfrom pathlib import Path\n'
        'work=Path("adaptive_mamba_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))\nwork.mkdir()\n'
        'files=json.loads(gzip.decompress(base64.b64decode('+repr(payload())+')))\n'
        'for name,encoded in files.items():\n    destination=work/name\n    destination.parent.mkdir(parents=True,exist_ok=True)\n    destination.write_bytes(base64.b64decode(encoded))\n'
        'for document,field in [("protocol.json","frozen_files"),("evaluation_seal.json","files")]:\n'
        '    manifest=json.loads((work/"adaptive_router_experiment"/document).read_text())\n'
        '    for name,expected in manifest[field].items():\n        assert hashlib.sha256((work/name).read_bytes()).hexdigest()==expected,name\n'
        'print("Verified frozen inputs:",work)\n')
    add('markdown','''## Phase 1 — Reproduce CPU router training

Fit on the saved 1,000 training records and select epochs/threshold using only the 200 validation records. This runs in a separate folder and checks the fitted coefficients against the frozen router; evaluation uses the frozen copy. No benchmark labels are used for fitting.
''')
    add('code','training=work/"training_replay"\n(training/"adaptive_router_experiment").mkdir(parents=True)\n'
        'for name in ["train_adaptive_router.py","adaptive_memory_router.py","dual_state_mamba.py","adaptive_router_experiment/dataset.json"]:\n    shutil.copyfile(work/name,training/name)\n'
        'subprocess.run([sys.executable,"train_adaptive_router.py"],cwd=training,check=True)\n'
        'trained=json.loads((training/"adaptive_router_experiment/router.json").read_text())\n'
        'frozen=json.loads((work/"adaptive_router_experiment/router.json").read_text())\n'
        'for head in ["fast","detailed"]:\n    assert max(abs(a-b) for a,b in zip(trained[head]["weights"],frozen[head]["weights"]))<1e-9\n    assert abs(trained[head]["bias"]-frozen[head]["bias"])<1e-9\n'
        'assert trained["threshold"]==frozen["threshold"]\nprint("Router training reproduced.")\n')
    add('markdown','''## Phase 2 — Run the ordered GPU comparison

Confirm CUDA and one plain generation; measure original clean/safe/attacked accuracy; stop if no attack drop; check all state tensors; evaluate adaptive dual state; run the observed-set regression; measure synchronized speed. The regression original-model accuracy is reused from the saved historical run, while the independent benchmark and timing run both variants here. The original HiSPA extractor supplies state access.

Candidate ranking and free-generation accuracy are separate. The latter uses the unchanged strict value-at-start parser. Timing uses 20 notes × 3 repeats × 2 variants and includes the full routing/prefill/generation path. Mamba weights remain frozen and their hash is rechecked.
''')
    add('code','subprocess.run([sys.executable,"-u","evaluate_adaptive_mamba.py"],cwd=work,check=True)\n')
    add('markdown','## Phase 3 — Read results and save artifacts\n')
    add('code','status=json.loads((work/"adaptive_router_experiment/status.json").read_text())\nprint(status)\n'
        'assert status["status"]=="complete","Inspect the attack gate before proceeding."\n'
        'subprocess.run([sys.executable,"write_adaptive_report.py"],cwd=work,check=True)\n'
        'from IPython.display import Markdown,display,FileLink\n'
        'display(Markdown((work/"adaptive_router_experiment/REPORT.md").read_text()))\n'
        'archive=shutil.make_archive(str(work),"zip",root_dir=work)\ndisplay(FileLink(archive))\n')
    result_path=ROOT/'adaptive_router_experiment/results.json'
    if result_path.exists():
        r=json.loads(result_path.read_text())
        text=['## Recorded result','','| Variant | Clean ranked | Attacked ranked | Safe ranked | Full tokens/sec |','| --- | ---: | ---: | ---: | ---: |']
        for variant,s in r['summary'].items():
            text.append('| '+variant+' | '+' | '.join(f"{s[c]['candidate']['accuracy']:.1%}" for c in ['clean','malicious','safe'])+f" | {s['timing']['end_to_end_tokens_per_second']:.2f} |")
        text+=['','These are recorded measurements. Rerunning reproduces the same test, not a new independent dataset.']
        add('markdown','\n'.join(text)+'\n')
    notebook={'nbformat':4,'nbformat_minor':5,'cells':cells,'metadata':{'accelerator':'GPU',
              'kernelspec':{'name':'python3','language':'python','display_name':'Python 3'},
              'colab':{'name':'adaptive_mamba.ipynb','provenance':[]}}}
    (ROOT/'adaptive_mamba.ipynb').write_text(json.dumps(notebook,indent=1)+'\n')
    print('Built adaptive_mamba.ipynb with',len(FILES),'embedded files.')


if __name__=='__main__':build()
