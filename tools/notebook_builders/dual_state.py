"""Package the completed synthetic dataset and experiment as one Colab notebook."""
import base64
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "dual_state_run"


def build():
    cells = []

    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})

    def code(text):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": text.splitlines(keepends=True)})

    md("""# Frozen Mamba-130m: two memory banks and a learned router

Run this notebook top to bottom in a Colab **T4 GPU** runtime. It compares exactly:
1. Original pretrained `state-spaces/mamba-130m-hf`.
2. The same frozen weights with separate trusted/candidate memory banks and a small text router.

Only the 8,193-parameter router is trained. No backbone fine-tuning, new model download beyond Mamba-130m, or automatic package installation is performed.

The 1,000-example dataset is embedded below. Its external spans were already generated locally using Ollama `llama3.2:3b` at temperature **0.7**, with no paid API calls. Source facts/questions/gold answers are controlled programmatically. This is a synthetic prompt-injection pilot, not a reproduction of optimized HiSPA triggers.

The original HiSPA extractor is embedded verbatim in the script with its checksum. Compatibility changes adapt its cache API. All 48 convolution/SSM tensors must be accessible before defense evaluation proceeds.
""")

    md("## Phase A — Experiment source\n\nThis cell writes the complete runnable script. Its CLI can also generate data on a Mac with the existing Ollama model: `python3 dual_state_mamba.py generate`.\n")
    code("%%writefile dual_state_mamba.py\n" + (ROOT / "dual_state_mamba.py").read_text())

    files = {name: (RUN / name).read_text() for name in [
        "synthetic_data.json", "generation_config.json", "generation_summary.json",
        "data_corrections.json", "data_reviews.json",
    ]}
    payload = base64.b64encode(gzip.compress(json.dumps(files).encode(), mtime=0)).decode()
    md("""## Phase B — Load and audit the generated dataset

There are 500 safe and 500 malicious records. The 700/200/100 split keeps source pairs together and holds out two attack-generation families. One hundred safe records need information in the external span. A documented factual correction is included; raw generation and its journal are retained in the project folder.

Automatic checks do not establish complete semantic validity or attack effectiveness.
""")
    code("import base64, gzip, json, runpy\nfrom pathlib import Path\nfrom types import SimpleNamespace\n"
         "experiment = runpy.run_path('dual_state_mamba.py', run_name='dual_state_experiment')\n"
         "run_dir = Path('dual_state_run')\nrun_dir.mkdir(exist_ok=True)\n"
         "embedded = json.loads(gzip.decompress(base64.b64decode(" + repr(payload) + ")))\n"
         "for name, text in embedded.items():\n    (run_dir / name).write_text(text)\n"
         "data = json.loads((run_dir / 'synthetic_data.json').read_text())\n"
         "assert data['complete'] and len(data['records']) == 1000\n"
         "audit = experiment['audit_dataset'](run_dir)\nassert audit['unresolved_flag_count'] == 0\n"
         "experiment['table'](['Split', 'Safe', 'Malicious'], [[split,\n"
         "    sum(r['split'] == split and r['attack_label'] == 'safe' for r in data['records']),\n"
         "    sum(r['split'] == split and r['attack_label'] == 'malicious' for r in data['records'])]\n"
         "    for split in ['train', 'validation', 'test']])\n")

    md("""## Phase C — Train only the router

The router is a CPU logistic classifier over hashed word/bigram features of the external span. Test labels, answers, and attack-family names never enter its features. Checkpoint and threshold selection use validation data only. The printed accuracy measures routing, not Mamba retrieval.
""")
    code("args = SimpleNamespace(output=str(run_dir), max_test=0)\nexperiment['train_router'](args)\n")

    md("""## Phase D — GPU setup and one plain-prompt generation

This stops if CUDA or required packages are unavailable. Tested with PyTorch 2.11.0+cu130 and Transformers 5.17.0. The model uses float32. No dependencies are installed automatically. The smoke-test time includes startup effects and is not the final speed benchmark.
""")
    code("import torch, transformers\nassert torch.cuda.is_available(), 'Select Runtime → Change runtime type → T4 GPU.'\n"
         "print('Versions:', torch.__version__, transformers.__version__)\n"
         "model, tokenizer = experiment['load_mamba']()\n"
         "assert all(not p.requires_grad for p in model.parameters())\n")

    md("""## Phase E — Reproduce an accuracy drop on validation data

Compare clean versus injected versions of the same 20 preselected validation scenarios. Both use the same forced choice between single-token **True** and **False**. This explicitly measures constrained-choice retrieval and avoids scoring every free-form response invalid. If no accuracy drop occurs, execution stops here.
""")
    code("gate = experiment['baseline_gate'](model, tokenizer, run_dir)\n"
         "assert gate['attack_drop_observed'], 'No attack drop: revise attacks using development data before defense claims.'\n")

    md("""## Phase F — Check state isolation, then evaluate both models

Checks compare the cached chunk continuation with both original token-by-token processing and full prefill. They verify independent storage, unchanged trusted memory, restoration, and `.generate()` handoff. The continuation adds the initial state's mathematically required contribution to the existing zero-initialized scan; pretrained weights are unchanged.

The test evaluates 50 source pairs under clean, safe-note, and malicious-note conditions. Both variants see identical prompts and decoding rules. The router alone chooses the memory bank; test attack labels are used only for scoring.

Speed uses 16-token generations on six preselected test records with three repetitions, after warm-up. CUDA is synchronized around `.generate()` and the complete inference path. **Full tokens/sec** includes routing, state extraction/cloning, candidate processing, question prefill, and generation. Generation-only speed excludes that defense work and must not be presented as overall speed.
""")
    code("results = experiment['evaluate'](args, model, tokenizer)\n"
         "assert results['backbone_before_sha256'] == results['backbone_after_sha256']\n")

    md("## Phase G — Interpret the result and save artifacts\n\nA defense must preserve useful retrieval, not merely restore a constant answer. The checks below flag that failure mode explicitly.\n")
    code("from collections import Counter\n"
         "for variant in ['original', 'dual_state']:\n"
         "    clean = [r for r in results['results'] if r['variant'] == variant and r['condition'] == 'clean']\n"
         "    distribution = Counter(r['prediction'] for r in clean)\n"
         "    print(variant, 'clean prediction distribution:', dict(distribution))\n"
         "    if len(distribution) == 1:\n"
         "        print('LIMITATION: constant clean predictions; this does not demonstrate successful comprehension.')\n"
         "base_params = results['state_checks']['backbone_parameters']\n"
         "experiment['table'](['Model', 'Total parameters', 'Backbone trained'], [\n"
         "    ['Original', base_params, 0], ['Dual state + router', base_params + 8193, 0]])\n"
         "print('Detailed records, timings, checks, and weight hashes:', run_dir / 'mamba_results.json')\n")

    recorded = RUN / "mamba_results.json"
    if recorded.exists():
        r = json.loads(recorded.read_text())
        lines = ["## Recorded research result (2026-09-23)", "",
                 "These values come from the completed run saved with this project. Running the cells above produces a new measurement.", "",
                 "| Model | Clean | Attacked | End-to-end tokens/sec |", "| --- | ---: | ---: | ---: |"]
        for variant, values in r["summary"].items():
            lines.append(f"| {variant} | {values['clean']['accuracy']:.0%} | {values['malicious']['accuracy']:.0%} | {values['end_to_end_tokens_per_second']:.2f} |")
        lines.extend(["", "Both models predicted True on every clean test item. Therefore the observed recovery to 50% is restoration of a constant answer, not evidence of useful retrieval robustness. Router test accuracy was 98%, which is a separate classification result."])
        md("\n".join(lines) + "\n")

    for index, cell in enumerate(cells):
        cell["id"] = f"phase-cell-{index:02d}"
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
                "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}, "colab": {"name": "dual_state_mamba.ipynb", "provenance": []}}}
    output = ROOT / "dual_state_mamba.ipynb"
    output.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"Wrote {output.name}: {len(cells)} cells, {output.stat().st_size:,} bytes")


if __name__ == "__main__":
    build()
