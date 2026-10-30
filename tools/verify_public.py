"""Read-only public-export checks; never a substitute for the full internal gate."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def require(value, reason):
    if not value:
        raise ValueError(reason)


def main():
    manifest = json.loads((ROOT / 'provenance/public_manifest.json').read_text())['files']
    actual={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()
            and '.git' not in p.relative_to(ROOT).parts and '__pycache__' not in p.parts
            and p.name!='.DS_Store' and p.suffix!='.pyc'
            and str(p.relative_to(ROOT))!='provenance/public_manifest.json'}
    require(actual==set(manifest),'Public manifest inventory coverage differs')
    for name in manifest:
        require(not (name.startswith('experiments/19_embedding_followup/raw/') and name!='experiments/19_embedding_followup/raw/provenance.json'),'Raw new BIPIA corpus in public export')
        require(name not in ['experiments/19_embedding_followup/prepared.json','experiments/19_embedding_followup/derived_official_test.jsonl'],'Raw prepared corpus in public export')
        require(not name.startswith(('experiments/20_sst2_followup/progress_snapshots/','experiments/20_sst2_followup/continuation_progress/')),'Raw response snapshot in public export')
        require(name not in ['experiments/20_sst2_followup/'+n for n in ['prepared.json','masks.json','screen_results.json','final_results.json','continuation_results.json','continuation_overlap.json']],'Raw new source/response material in public export')
    for name, expected in manifest.items():
        path = ROOT / name
        require(path.is_file() and not path.is_symlink(), 'Missing/nonmaterialized export: ' + name)
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'Public source changed: ' + name)
    forbidden = ['.codex', '.playwright-mcp', 'private', '.env', 'reviewer_packet.json',
                 'model_review_requests.jsonl', 'mapping.json']
    for name in manifest:
        path = Path(name)
        require(not set(path.parts) & set(forbidden) and path.suffix not in ['.zip', '.ipynb', '.log'],
                'Private/embedded payload in export: ' + name)
        if path.suffix == '.py':
            tree = ast.parse((ROOT / path).read_text())
            require(not any(isinstance(n, ast.Constant) and isinstance(n.value, str)
                            and n.value.startswith('H4sIA') and len(n.value) > 2000 for n in ast.walk(tree)),
                    'Embedded source/data payload in export: ' + name)
        if path.suffix == '.md':
            for raw in re.findall(r'\[[^\]]*\]\(([^)]+)\)', (ROOT / path).read_text()):
                if re.match(r'^[a-z]+:|^#', raw):
                    continue
                target = raw.split('#')[0].strip('<>')
                require((ROOT / path.parent / target).exists(), 'Broken public link: ' + name + ': ' + raw)
    spec = importlib.util.spec_from_file_location('public_task_scores', ROOT / 'src/evaluation/published_task_scoring.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = json.loads((ROOT / 'experiments/17_external_validation/results_v2.json').read_text())
    counts = {}
    for task in ['sst2', 'sms_spam', 'mrpc']:
        rows = [r for r in result['development'] if r['dataset'] == task]
        require(len(rows) == 24 and len({r['source_id'] for r in rows}) == 24, 'Wrong public response coverage')
        for row in rows:
            strict, official = module.predictions(task, row['response'])
            require(row['strict'] == strict and row['official'] == official, 'Saved response score mismatch')
        correct = sum(r['strict'] == r['gold'] for r in rows)
        valid = sum(r['strict'] is not None for r in rows)
        empty = module.predictions(task, result['instruction_only'][task]['response'])[0]
        gain = (correct - sum(empty == r['gold'] for r in rows)) / 24
        expected = {'strict_correct': correct, 'strict_valid': valid, 'input_gain': gain,
                    'passed': correct >= 18 and valid >= 22 and gain >= .2}
        require(result['gate'][task] == expected, 'Public gate differs from included raw responses')
        counts[task] = {'strict_correct': correct, 'strict_valid': valid,
                        'published_correct': sum(r['official'] == r['gold'] for r in rows)}
    require(result['status'] == 'stopped_clean_gate' and not result['gate_passed'] and not result['final'],
            'External failure must not become an efficacy result')
    require(result['backbone_before'] == result['backbone_after'], 'External terminal weight check failed')
    timing = json.loads((ROOT / 'experiments/18_runtime_validation/timing_results.json').read_text())
    require(timing['status'] == 'failed' and 'CUDA out of memory' in timing['error']
            and len(timing['measurements']) == 166 and len(timing['warmups']) == 36
            and 'backbone_after' not in timing, 'Full timing failure is misstated')
    for length in [128, 512]:
        groups = {mode: {r['repetition']: r for r in timing['measurements']
                         if r['prefix_tokens'] == length and r['mode'] == mode}
                  for mode in ['original', 'forced_accept', 'forced_reject', 'identical_decision_filtering']}
        require(all(set(g) == set(range(20)) for g in groups.values()), 'Short timing block incomplete')
        for left, right in [('original', 'forced_accept'), ('forced_reject', 'identical_decision_filtering')]:
            require(all(groups[left][rep]['generated_tokens'] == groups[right][rep]['generated_tokens']
                        for rep in range(20)), 'Observed short-block output mismatch')
    sys.path.insert(0,str(ROOT/'tools'))
    from verify_followup_public import verify as verify_followup
    followup = verify_followup(ROOT)
    print(json.dumps({'status': 'passed', 'exported_files_checked': len(manifest),
                      'external_saved_response_scores': counts,
                      'timing': 'Full protocol failed at 166/240; complete short blocks have 80 matching output pairs',
                      'followup':followup,
                      'scope': 'Public bytes, syntax, links and included saved-output checks only. No withheld-source verification, private annotation audit or fresh inference.'}, indent=2))


if __name__ == '__main__':
    main()
