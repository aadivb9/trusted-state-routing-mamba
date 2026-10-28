"""Package the frozen development gate without exposing final test examples."""
import base64
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = [
    "dual_state_mamba.py", "external_benchmark/run_competence.py",
    "external_benchmark/PROTOCOL.md", "external_benchmark/competence_seal.json",
    "external_benchmark/prepared/development.json", "external_benchmark/prepared/audit.json",
    "external_benchmark/upstream/OpenPromptInjection/evaluator/utils.py",
]


def main():
    files = {name: (ROOT / name).read_text() for name in NAMES}
    hashes = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()}
    payload = base64.b64encode(gzip.compress(json.dumps(files).encode(), mtime=0)).decode()
    setup = f'''import base64, gzip, hashlib, json, os, pathlib, subprocess, sys
root = pathlib.Path('/content/mamba_external_development_20261010')
root.mkdir(exist_ok=True)
files = json.loads(gzip.decompress(base64.b64decode({payload!r})))
expected = {hashes!r}
for name, content in files.items():
    assert hashlib.sha256(content.encode()).hexdigest() == expected[name]
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text() == content, 'Do not replace an existing experiment'
    path.write_text(content)
print('Prepared development gate; final benchmark examples are not included')
'''
    run = '''result_dir = root / 'development_run_01'
assert not result_dir.exists(), 'Use a fresh run directory; preserve previous evidence'
with open(root / 'development_run_01.log', 'w') as log:
    completed = subprocess.run([sys.executable, '-u', str(root / 'external_benchmark/run_competence.py'), '--output', str(result_dir)], cwd=root, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'HF_HUB_DISABLE_PROGRESS_BARS':'1', 'TOKENIZERS_PARALLELISM':'false'})
print('EXIT', completed.returncode)
print((root / 'development_run_01.log').read_text()[-5000:])
if (result_dir / 'results.json').exists():
    result_bytes = (result_dir / 'results.json').read_bytes()
    print('RESULT_GZIP_B64', base64.b64encode(gzip.compress(result_bytes)).decode())
'''
    cells = [{"cell_type": "markdown", "metadata": {}, "source": [
        "# External benchmark: Mamba-2.8B development gate\n",
        "Run after validating T4/cache feasibility. Uses 72 separate development examples, frozen FP32 weights and published instructions. No package installation. Final benchmark examples are deliberately absent. A failed competence gate stops the planned final evaluation.\n"]}]
    for code in [setup, run]:
        cells.append({"cell_type": "code", "metadata": {}, "source": code.splitlines(keepends=True),
                      "execution_count": None, "outputs": []})
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"}},
                "cells": cells}
    (ROOT / "external_benchmark_competence.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
    print("Wrote external_benchmark_competence.ipynb")


if __name__ == "__main__":
    main()
