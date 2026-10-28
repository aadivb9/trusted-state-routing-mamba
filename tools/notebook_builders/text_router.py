"""Package the sealed holdout and frozen implementation as one Colab notebook."""
import base64
import gzip
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
NAMES = ['dual_state_mamba.py','clean_baseline_diagnostic.py','cloze_dual_state_eval.py',
         'dual_state_run/router.json','fresh_robustness_eval.py','fresh_robustness_data.py',
         'fresh_robustness_run/protocol.json','fresh_robustness_run/dataset.json',
         'fresh_robustness_run/seal.json','fresh_robustness_run/semantic_audit.json',
         'fresh_robustness_run/data_corrections.json','fresh_robustness_run/generation_accounting.json']


def build():
    # Preserve exact historical bytes and their original internal filenames.
    with zipfile.ZipFile(ROOT/'text_router_benchmark_archive.zip') as archive:
        files = {name:archive.read(name).decode() for name in NAMES}
    payload = base64.b64encode(gzip.compress(json.dumps(files).encode(),mtime=0)).decode()
    cells = []

    def markdown(text):
        cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)})

    def code(text):
        cells.append({'cell_type':'code','metadata':{},'source':text.splitlines(keepends=True),
                      'execution_count':None,'outputs':[]})

    markdown('''# Mamba: text-only router benchmark

Run top to bottom on a Colab **T4 GPU**. This notebook embeds a new 200-record test set (100 safe / 100 malicious), the existing fitted router, and the exact frozen implementation. There is no new training or paid generation in this notebook. The only downloaded model is `state-spaces/mamba-130m-hf`. No packages are installed automatically. The implementation was previously tested with PyTorch 2.11.0+cu128 and Transformers 5.16.1; the recorded status below distinguishes completed measurements from an interrupted run.

The original and dual-state variants share the same frozen 130m backbone. The added text router has 8,193 parameters. The external-note boundary is known. The original HiSPA extractor remains embedded in the frozen experiment source.

New source facts, paraphrased contexts, subtle false research notes, and suspicious-looking safe notes were generated without Mamba or router feedback. Claude Code's signed-in subscription generated the spans and performed a separate label-blinded semantic audit. Its CLI does not expose temperature, so no 0.7 claim applies to this new set. One safe-note factual error was corrected against the controlled source before sealing. The original 1,000-example Ollama set remains separate.

The primary comparison has exactly two model variants. A final diagnostic checks whether removing spans rejected by the same router gives the same result as restoring memory; it introduces no additional trained model.
''')
    markdown('## Phase 1 — Restore the sealed experiment into a new folder\n\nEach execution uses a fresh directory so previous results cannot be mistaken for a new run.\n')
    code('import base64,gzip,json,hashlib,datetime,subprocess,sys\nfrom pathlib import Path\n'
         'work=Path("text_router_reproduction_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))\n'
         'work.mkdir()\n'
         'embedded=json.loads(gzip.decompress(base64.b64decode('+repr(payload)+')))\n'
         'for name,content in embedded.items():\n    destination=work/name\n    destination.parent.mkdir(parents=True,exist_ok=True)\n    destination.write_text(content)\n'
         'print("Experiment directory:",work)\n')
    markdown('## Phase 2 — Verify data and frozen implementation\n\nThe hashes were fixed before model evaluation. Neither router threshold nor prompts are tuned on this set.\n')
    code('seal=json.loads((work/"fresh_robustness_run/seal.json").read_text())\n'
         'for name,expected in seal["frozen_files"].items():\n    assert hashlib.sha256((work/name).read_bytes()).hexdigest()==expected\n'
         'assert hashlib.sha256((work/"fresh_robustness_run/dataset.json").read_bytes()).hexdigest()==seal["dataset_sha256"]\n'
         'assert hashlib.sha256((work/"fresh_robustness_eval.py").read_bytes()).hexdigest()==seal["evaluation_source_sha256"]\n'
         'print("All frozen hashes match.")\n')
    markdown('''## Phase 3 — Execute the ordered GPU experiment

The script first confirms CUDA and runs one plain `.generate()`. It evaluates original-model clean, safe, and attacked cases, using both candidate ranking and freely generated answers. If candidate accuracy does not drop under attack, defense evaluation stops. It then checks state access/isolation and evaluates dual state on the same cases.

Accuracy from candidate ranking and accuracy from free generation are reported separately. Free answers use a fixed value-at-start parser; invalid answers count as wrong. All 100 source groups are retained regardless of outcome.

The speed benchmark uses 20 fixed records × 3 repetitions × 2 variants, with 16 generated tokens, warm-up, alternating order, and CUDA synchronization. Both variants explicitly request only last-token logits during generation. Full timing includes tokenization, routing, extraction, cloning, external-note processing, prefill, and generation. Candidate-scoring throughput is measured separately. Memory figures are CUDA allocated memory; CPU snapshots are not included.
''')
    code('subprocess.run([sys.executable,"-u","fresh_robustness_eval.py"],cwd=work,check=True)\n')
    markdown('## Phase 4 — Inspect results, routing errors, and the filtering diagnostic\n\nPaired confidence intervals resample source groups. A result on these synthetic notes does not establish optimized HiSPA resistance or general natural-abstract comprehension.\n')
    code('run_dir=work/"fresh_robustness_run"\n'
         'status=json.loads((run_dir/"status.json").read_text())\n'
         'print("Status:",status)\n'
         'assert status["status"]=="complete", "Stopped at the attack gate; inspect attack_gate.json."\n'
         'results=json.loads((run_dir/"results.json").read_text())\n'
         'assert results["manifest"]["backbone_before_sha256"]==results["backbone_after_sha256"]\n'
         'for name in ["summary","router_metrics","paired_differences","filtering_equivalence","success_criteria"]:\n    print(name,json.dumps(results[name],indent=2))\n')
    markdown('## Phase 5 — Save the reproduction artifacts\n')
    code('import shutil\nfrom IPython.display import FileLink,display\n'
         'archive=shutil.make_archive(str(work),"zip",root_dir=work)\n'
         'display(FileLink(archive))\n')
    recorded = ROOT/'text_router_benchmark/results.json'
    if not recorded.exists():
        markdown('## Current execution status\n\nThe dataset and router evaluation are complete locally. The prior GPU runtime was interrupted, and a complete fresh-test GPU result has not been recovered. Running this notebook performs the sealed evaluation from the beginning; the earlier 94% attacked accuracy belongs to a different, reused test set.\n')
    if recorded.exists():
        results = json.loads(recorded.read_text())
        lines = ['## Recorded run','',
                 'These are saved measurements. Running the notebook creates a new reproduction, not a new independent test set.','',
                 '| Variant | Clean ranked | Attacked ranked | Safe ranked | Attacked free | Full tokens/s |',
                 '| --- | ---: | ---: | ---: | ---: | ---: |']
        for variant,summary in results['summary'].items():
            lines.append('| '+variant+' | '+' | '.join(f"{summary[condition]['candidate']['accuracy']:.1%}" for condition in ['clean','malicious','safe'])+
                         f" | {summary['malicious']['free_generation']['accuracy']:.1%} | {summary['timing']['end_to_end_tokens_per_second']:.2f} |")
        lines += ['', 'Frozen primary criteria: '+json.dumps(results['success_criteria']),
                  '', 'See `text_router_benchmark/REPORT.md` in the project for interpretation and limits.']
        markdown('\n'.join(lines)+'\n')
    for index,cell in enumerate(cells):
        cell['id']=f'fresh-robustness-{index:02d}'
    notebook = {'nbformat':4,'nbformat_minor':5,'cells':cells,
                'metadata':{'accelerator':'GPU','kernelspec':{'name':'python3','language':'python','display_name':'Python 3'},
                            'language_info':{'name':'python'},'colab':{'name':'text_router_benchmark.ipynb','provenance':[]}}}
    output = ROOT/'text_router_benchmark.ipynb'
    output.write_text(json.dumps(notebook,indent=1)+'\n')
    print('Wrote',output.name,len(cells),'cells',output.stat().st_size,'bytes')


if __name__=='__main__':
    build()
