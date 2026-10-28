"""Export an explicit, hashed code-review bundle as plain text; no uploads."""
import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'code_review_20261016'
SOURCES = [
    'adaptive_memory_router.py', 'train_adaptive_router.py',
    'dual_state_mamba.py', 'cloze_dual_state_eval.py',
    'clean_baseline_diagnostic.py', 'evaluate_adaptive_mamba.py',
    'evaluate_text_router_benchmark.py', 'build_generalization_data.py',
    'evaluate_generalization.py', 'evaluate_generalization_boundary_fix.py',
    'final_hispa_audit/evaluate.py', 'final_hispa_audit/summarize.py',
    'external_benchmark/prepare_data.py', 'external_benchmark/run_competence.py',
    'external_benchmark/audit_development.py', 'external_benchmark/verify_preparation.py',
    'research_closeout/run_external_final.py', 'research_closeout/router_coverage.py',
    'research_closeout/resume_external_final.py', 'research_closeout/audit_final.py',
    'test_adaptive_memory_router.py', 'test_generalization.py',
    'test_generalization_boundary_fix.py', 'write_generalization_report.py',
    'review_followup_20261016/audit_hispa.py',
    'review_followup_20261016/package_code.py',
]
DOCS = [
    'final_hispa_audit/protocol.json', 'final_hispa_audit/source_provenance.json',
    'final_hispa_audit/REPORT.md', 'final_hispa_audit/verification.json',
    'external_benchmark/PROTOCOL.md', 'external_benchmark/upstream_manifest.json',
    'research_closeout/PROTOCOL.md', 'research_closeout/RESUME_AMENDMENT.md',
    'research_closeout/REPORT.md', 'research_closeout/run_03/verification.json',
    'adaptive_router_experiment/router.json', 'dual_state_run/router.json',
    'paper/evidence_ledger.md', 'paper/build_requirements.txt',
    'review_followup_20261016/HISPA_REVIEW.md',
    'review_followup_20261016/NEXT_EXPERIMENT.md',
    'review_followup_20261016/hispa_audit.json',
    'review_followup_20261016/control_tests.log',
]
CORE = ['adaptive_memory_router.py', 'train_adaptive_router.py',
        'final_hispa_audit/evaluate.py', 'cloze_dual_state_eval.py',
        'external_benchmark/run_competence.py', 'dual_state_mamba.py']


def main():
    OUT.mkdir(exist_ok=True)
    exports = {}
    for source in SOURCES + DOCS:
        original = ROOT/source
        raw = original.read_bytes()
        name = source + '.txt'
        path = OUT/'files'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        exports['files/'+name] = dict(original=source, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    # Expose the compressed upstream source so reviewers need not decode it.
    tree = ast.parse((ROOT/'dual_state_mamba.py').read_text())
    constants = {n.targets[0].id:ast.literal_eval(n.value) for n in tree.body
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id in ['HISPA_SOURCE_GZIP_B64','HISPA_SOURCE_SHA256']}
    upstream = gzip.decompress(base64.b64decode(constants['HISPA_SOURCE_GZIP_B64']))
    assert hashlib.sha256(upstream).hexdigest() == constants['HISPA_SOURCE_SHA256']
    name = 'files/embedded_hispa_source.py.txt'
    (OUT/name).write_bytes(upstream)
    exports[name] = dict(original='Decoded verbatim from dual_state_mamba.py; upstream HiSPA source, not original project authorship',
                         sha256=hashlib.sha256(upstream).hexdigest(), bytes=len(upstream))
    readme = '''CODE REVIEW PACKAGE — October 16, 2026

Start with core_code.txt for the six main files in one uploadable text document.
The files/ directory contains individual plain-text copies with original paths.
Every .py.txt and .json.txt copy preserves the original bytes. manifest.json
records hashes. Strip the final .txt suffix when restoring files for execution.

READING ORDER
1. adaptive_memory_router.py: the four supported attributes, 14 features,
   fast/detailed logistic heads, and rejection decision.
2. train_adaptive_router.py: training/validation inputs and threshold selection.
3. final_hispa_audit/evaluate.py: undefended versus defended paths, triggers,
   empty-context and matched-length controls, and model input construction.
4. cloze_dual_state_eval.py: candidate_scores(), including causal score shift.
5. external_benchmark/run_competence.py: published prompts, clean-task gate,
   generated-token extraction and both scoring rules.
6. dual_state_mamba.py: text_features()/train_router() for the original hashed
   word/bigram baseline; install_chunk_continuation()/make_state_access() for
   the cache mechanics. Its earlier True/False evaluator is historical and is
   not the final HiSPA cloze evaluator.

The individual files also include dataset construction, generalization and
boundary repair, external preparation, final diagnostics, recovery, and tests.
The embedded upstream HiSPA source is decoded into its own file for inspection;
credit belongs to its original authors. make_state_access() shows adaptations.

WHAT THE REVIEW FOUND
- The undefended HiSPA pilot already scored 16/16 under every published trigger.
- A literal source-answer match solves 16/16 pilot and 32/32 parent tasks.
- The inspected inference path does not receive the gold index or attack label.
- The dataset's separate semantic reviewer always saw the correct answer first.
- On the same 16 sources, older synthetic late attacks scored 6/16 for both
  models. This is a retrospective subset using a different attack protocol.
Read files/review_followup_20261016/HISPA_REVIEW.md.txt for evidence and limits.

CURRENT VS PROPOSED
This is the evaluated implementation, not an embedding rewrite. Embeddings,
new held-out domains and fresh inference stress controls are proposed in
files/review_followup_20261016/NEXT_EXPERIMENT.md.txt. They have not been run.
The paper now labels the schema failure and benchmark competence failure
separately. The latter did not measure attack efficacy.

REPRODUCTION SCOPE
This is a code-review bundle, not a turnkey reproduction archive. It omits
model weights, full datasets, generation/API conversation logs and environments.
Evaluators expect their original project-root layout and recorded input/seal
files. GPU runs need compatible torch/transformers/numpy and CUDA; original
JSON results/notebooks record versions. paper/build_requirements.txt is only
for rendering the manuscript, not GPU evaluation dependencies. External
preparation imports the pinned upstream Open-Prompt-Injection repository; its
commit and file hashes are included. No new project-wide license is implied
for embedded upstream code or third-party data. No repository was published.

The existing full local closing evidence archive remains separate and unchanged.
'''
    (OUT/'README.txt').write_text(readme)
    parts = [readme, '\n\nCORE SOURCE FILES\n']
    for name in CORE:
        parts.extend(['\n'+'='*78+'\nSOURCE: '+name+'\n'+'='*78+'\n', (ROOT/name).read_text()])
    (OUT/'core_code.txt').write_text('\n'.join(parts))
    # Only source/docs are exported; additionally reject obvious credential literals.
    patterns = [r'\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}', r'\bgh[pousr]_[A-Za-z0-9]{20,}',
                r'\bAKIA[0-9A-Z]{16}', r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']
    for path in OUT.rglob('*.txt'):
        text = path.read_text()
        assert not any(re.search(pattern, text) for pattern in patterns), f'Credential-shaped literal in {path.name}'
    for name in ['README.txt','core_code.txt']:
        raw=(OUT/name).read_bytes()
        exports[name]=dict(original='Generated review guide/bundle',sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    manifest=dict(date='2026-10-16',scope='Local code review; no dataset release or public repository', files=exports)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=ROOT/'code_review_20261016.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in [*sorted(exports),'manifest.json']:
            z.write(OUT/name, 'code_review_20261016/'+name)
    with zipfile.ZipFile(archive) as z:
        for name, info in exports.items():
            assert hashlib.sha256(z.read('code_review_20261016/'+name)).hexdigest()==info['sha256']
    report=dict(status='passed',files_verified=len(exports),
                archive=archive.name,archive_bytes=archive.stat().st_size,
                archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                scope='Byte-identical source export and ZIP integrity; not execution of the exported GPU evaluators')
    (ROOT/'review_followup_20261016/package_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
