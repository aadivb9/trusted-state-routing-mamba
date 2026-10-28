"""Read-only evidence gate; only --output writes a check record.

Auditors that write reports run in disposable copies. No model inference.
"""
import argparse
import ast
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REV = 'revision_experiments'
EXPERIMENTS = ('hispa_stress', 'competence', 'router_transfer')
ENTRY_POINTS = ('README.md', 'FINAL_RESEARCH_SUMMARY.md', 'PAPER_INTRODUCTION.md',
                'research_closeout/STATUS.md', 'paper/README.md',
                'paper/plain_language_summary.md', 'paper/evidence_ledger.md',
                'revision_experiments/README.md')
REVIEWED = ENTRY_POINTS + ('paper/manuscript.md', 'revision_experiments/REPORT.md')
START, END = '<!-- FORGE_STATUS_START -->', '<!-- FORGE_STATUS_END -->'


def read(root, name):
    return json.loads((root / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def current_status(root, name):
    h = read(root, f'{REV}/hispa_stress/results.json')
    c = read(root, f'{REV}/competence/results.json')
    r = read(root, f'{REV}/router_transfer/results.json')
    for result in (h, c, r):
        require(result['status'] == 'complete', 'A follow-up is incomplete')
    require(c['gate_passed'] is False, 'Gate changed: re-review narrative scope')
    clean = h['summary']['clean']
    e = r['summary']['synthetic_plus_email/embeddings/table']
    url = os.path.relpath(root / REV / 'REPORT.md', (root / name).parent)
    return (f'{START}\n'
            f'**Current evidence status — experiments completed October 16, 2026:** '
            f'{len(h["rows"])} HiSPA evidence-control evaluations; '
            f'{len({x["model"] for x in r["rows"]})} frozen routers on '
            f'{len({x["id"] for x in r["rows"]}):,} held-out-domain presentations; '
            f'{len(c["rows"])} instruction-tuned Mamba development examples. '
            f'Clean HiSPA two-choice ranking is {clean["correct"]}/{clean["n"]}; '
            f'strict free-generation correctness is {clean["free_correct"]}/{clean["n"]} '
            f'(different scoring methods, not a measured attack-induced drop). '
            f'Expanded-data embeddings reject {e["recall"]:.0%} of table attacks '
            f'and {e["false_positive_rate"]:.0%} of benign tables. '
            f'The instruction-tuned checkpoint failed the fixed gate. '
            f'No new end-to-end defense-efficacy evaluation ran; '
            f'a general defense and a benefit unique to two banks remain unestablished. '
            f'[Completed evidence and limits]({url}). '
            f'Earlier dated status and proposal text below is historical.\n{END}')


def statuses(root):
    for name in ENTRY_POINTS:
        text = (root / name).read_text()
        blocks = re.findall(re.escape(START) + r'.*?' + re.escape(END), text, re.S)
        require(blocks == [current_status(root, name)], f'Stale/missing current status: {name}')


def paper_tables(root):
    text = (root / 'paper/manuscript.md').read_text()
    h = read(root, f'{REV}/hispa_stress/results.json')['summary']
    for condition, s in h.items():
        row = f'| {condition} | {s["correct"]}/{s["n"]} | {s["free_correct"]}/{s["n"]} |'
        require(text.count(row) == 1, f'HiSPA table mismatch: {condition}')
    r = read(root, f'{REV}/router_transfer/results.json')['summary']
    for regime, label in [('synthetic_only', 'Synthetic'), ('synthetic_plus_email', 'Synthetic + email')]:
        for rep in ('symbolic', 'bigrams', 'embeddings'):
            cells = [r[f'{regime}/{rep}/{d}'] for d in ('table', 'code')]
            values = ' | '.join(f'{s["recall"]:.1%} / {s["false_positive_rate"]:.1%}' for s in cells)
            require(text.count(f'| {label} | {rep} | {values} |') == 1, f'Router table mismatch: {regime}/{rep}')
    c = read(root, f'{REV}/competence/results.json')['summary']
    strict = ', '.join(str(c[t]['strict_correct']) + '/24' for t in ('sst2', 'sms_spam'))
    strict += ' and ' + str(c['mrpc']['strict_correct']) + '/24'
    require(f'strict correctness is {strict}' in text, 'Instruction-tuned competence prose mismatch')
    baseline = read(root, 'research_closeout/run_03/results.json')['summary']
    for role in ('target', 'injected'):
        for task in ('sst2', 'sms_spam', 'mrpc'):
            s = baseline[f'{role}/{task}']
            row = f'| {role.capitalize()} | {task} | {s["official_correct"]}/100 | {s["strict_correct"]}/100 | {s["strict_valid"]}/100 |'
            require(row in text, f'Closing baseline table mismatch: {role}/{task}')


def review_freshness(root):
    reviewed = read(root, '.forge/narrative_review.json')
    require(set(reviewed['files']) == set(REVIEWED), 'Narrative review coverage changed')
    for name, digest in reviewed['files'].items():
        require(sha(root / name) == digest, f'Narrative changed since semantic review: {name}')


def render_freshness(root):
    record = read(root, 'paper/build_verification.json')
    required_inputs = {'paper/manuscript.md', 'paper/build.py', 'paper/references.bib',
                       'paper/build_requirements.txt', 'adaptive_router_experiment/results.json',
                       'text_router_benchmark/results.json', 'generalization_experiment/results.json',
                       'mamba_28b_feasibility/fp32_run_02/results.json',
                       'research_closeout/run_03/results.json', 'research_closeout/run_03/verification.json'}
    required_outputs = {'paper/manuscript.pdf', 'paper/manuscript.html', 'paper/figure_data.csv',
                        'paper/final_baseline_table.md'}
    require(set(record['inputs']) == required_inputs, 'Render input coverage changed')
    require(required_outputs <= set(record['outputs']), 'Missing rendered outputs')
    for name, digest in {**record['inputs'], **record['outputs']}.items():
        require(sha(root / name) == digest, f'Stale render/source: {name}')


def archive_parity(root):
    manifest_name = f'{REV}/evidence_manifest.json'
    manifest = read(root, manifest_name)['files']
    verification = read(root, f'{REV}/package_verification.json')
    archive = root / 'revision_experiments_evidence.zip'
    require(sha(archive) == verification['archive_sha256'], 'ZIP differs from verification hash')
    require(len(manifest) == verification['files_verified'], 'Manifest count differs')
    with zipfile.ZipFile(archive) as z:
        require(len(z.namelist()) == len(set(z.namelist())), 'Duplicate ZIP members')
        require(set(z.namelist()) == set(manifest) | {manifest_name}, 'Unexpected/missing ZIP members')
        require(z.read(manifest_name) == (root / manifest_name).read_bytes(), 'ZIP manifest differs')
        for name, digest in manifest.items():
            require(sha(root / name) == digest, f'Package stale against source: {name}')
            require(hashlib.sha256(z.read(name)).hexdigest() == digest, f'ZIP bytes differ: {name}')


def notebook_parity(root):
    path = root / REV / 'gpu_reproduction.ipynb'
    record = read(root, f'{REV}/notebook_verification.json')
    require(sha(path) == record['notebook_sha256'], 'Notebook hash differs')
    tree = ast.parse(''.join(json.loads(path.read_text())['cells'][1]['source']))
    encoded = next(n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute) and n.func.attr == 'b64decode')
    files = json.loads(gzip.decompress(base64.b64decode(encoded)))
    require(len(files) == record['embedded_files'], 'Notebook payload count differs')
    for name, content in files.items():
        require((root / name).read_bytes() == content.encode(), f'Notebook payload stale: {name}')


def source_provenance(root):
    base = root / REV / 'router_transfer/raw'
    provenance = json.loads((base / 'provenance.json').read_text())
    for name, digest in provenance['files'].items():
        require(sha(base / name) == digest, f'External raw source differs: {name}')


def run_audits(root, scripts):
    for script in scripts:
        p = subprocess.run([sys.executable, script], cwd=root, capture_output=True, text=True, timeout=90)
        require(p.returncode == 0, f'{script}: {p.stderr[-2500:]}')


def isolated_archive_audits(root):
    with tempfile.TemporaryDirectory(prefix='forge-package-') as tmp:
        dest = Path(tmp).resolve()
        with zipfile.ZipFile(root / 'revision_experiments_evidence.zip') as z:
            for name in z.namelist():
                require((dest / name).resolve().is_relative_to(dest), 'Unsafe ZIP member')
            z.extractall(dest)
        scripts = [f'{REV}/{e}/audit.py' for e in EXPERIMENTS]
        run_audits(dest, scripts)
        for e in EXPERIMENTS:
            name = f'{REV}/{e}/verification.json'
            require(read(dest, name) == read(root, name), f'Recomputed audit differs: {e}')


def current_evidence_audits(root):
    # Copy only source/data types needed by saved-evidence auditors, never archives/models.
    with tempfile.TemporaryDirectory(prefix='forge-current-') as tmp:
        dest = Path(tmp)
        for folder in (REV, 'research_closeout', 'external_benchmark', 'final_hispa_audit',
                       'text_router_benchmark', 'adaptive_router_experiment', 'generalization_experiment',
                       'mamba_28b_feasibility'):
            for p in (root / folder).rglob('*'):
                if p.is_file() and p.suffix in ('.py', '.json', '.jsonl', '.md') and '__pycache__' not in p.parts:
                    target = dest / p.relative_to(root)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, target)
        for p in root.glob('*.py'):
            shutil.copy2(p, dest / p.name)
        for p in (root / REV).rglob('*.before_tokenizer_fix'):
            shutil.copy2(p, dest / p.relative_to(root))
        scripts = [f'{REV}/{e}/audit.py' for e in EXPERIMENTS]
        scripts += ['research_closeout/audit_existing.py', 'research_closeout/audit_final.py']
        run_audits(dest, scripts)
        for e in EXPERIMENTS:
            name = f'{REV}/{e}/verification.json'
            require(read(dest, name) == read(root, name), f'Stale verification: {e}')
        for name in ('research_closeout/existing_results_verification.json', 'research_closeout/run_03/verification.json'):
            require(read(dest, name) == read(root, name), f'Stale historical verification: {name}')


CHECKS = (statuses, paper_tables, review_freshness, render_freshness, archive_parity,
          notebook_parity, source_provenance, current_evidence_audits, isolated_archive_audits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    checks = []
    for check in CHECKS:
        try:
            check(root)
            checks.append(dict(check=check.__name__, passed=True))
        except Exception as exc:
            checks.append(dict(check=check.__name__, passed=False, reason=str(exc)))
    input_names = list(REVIEWED) + [f'{REV}/{e}/{f}.json' for e in EXPERIMENTS for f in ('results', 'protocol')]
    input_names += ['paper/build_verification.json', f'{REV}/evidence_manifest.json',
                    'revision_experiments_evidence.zip', '.forge/narrative_review.json']
    result = dict(status='passed' if all(c['passed'] for c in checks) else 'failed', checks=checks,
                  checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  command='python3 .forge/check.py', checker_sha256=sha(Path(__file__)),
                  headline_inputs={n: sha(root/n) for n in input_names if (root/n).is_file()},
                  scope='Saved-output reanalysis, report/table consistency, provenance, output freshness and isolated ZIP audits; no new inference or guarantee of semantic completeness.')
    content = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    print(content)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
