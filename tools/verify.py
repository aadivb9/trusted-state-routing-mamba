"""Verify relocation, review views, documentation links, and the existing forge gate."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, reason):
    if not value:
        raise ValueError(reason)


def included(path):
    return '__pycache__' not in path.parts and path.name != '.DS_Store' and path.suffix != '.pyc'


def layout(root):
    return json.loads((root / 'docs/layout.json').read_text())


def preserved_bytes(root):
    spec = layout(root)
    inventory = json.loads((root / spec['original_inventory']).read_text())['files']
    base = root / spec['archive_root']
    for name, expected in inventory.items():
        require(sha(base / name) == expected, f'Original file changed during relocation: {name}')
    return {'original_files_checked': len(inventory)}


def review_views(root):
    spec = layout(root)
    base = root / spec['archive_root']
    files = 0
    for name, item in spec['views'].items():
        view, source = root / name, base / item['original']
        if item['kind'] == 'file':
            require(view.is_file(), f'Missing review source: {name}')
            require(sha(view) == sha(source), f'Review copy differs from preserved source: {name}')
            files += 1
        else:
            require(view.is_dir(), f'Missing review evidence directory: {name}')
            expected = {str(p.relative_to(source)) for p in source.rglob('*') if p.is_file() and included(p)}
            found = {str(p.relative_to(view)) for p in view.rglob('*') if p.is_file() and included(p)}
            require(found == expected, f'Review directory contents differ: {name}')
            for relative in expected:
                require(sha(view / relative) == sha(source / relative), f'Review evidence differs: {name}/{relative}')
                files += 1
    for name,item in spec.get('active_versions',{}).items():
        path=root/name
        require(path.exists() and not path.is_symlink(),f'Active version is missing or still aliases sealed evidence: {name}')
        if item.get('based_on'):require((base/item['based_on']).exists(),f'Active provenance target missing: {name}')
    return {'named_views': len(spec['views']), 'view_file_comparisons': files,
            'active_versions':len(spec.get('active_versions',{}))}


def source_coverage(root):
    spec = layout(root)
    inventory = json.loads((root / spec['original_inventory']).read_text())['files']
    expected = {n for n in inventory if '/' not in n and n.endswith(('.py', '.ipynb'))}
    found = {v['original'] for v in spec['views'].values()}
    require(expected <= found, f'Original top-level code/notebooks lack named views: {sorted(expected-found)}')
    for name in {n.split('/')[0] for n in inventory} - {'README.md', 'paper', '.forge'}:
        require(not (root / name).exists(), f'Unorganized historical item at root: {name}')
    return {'top_level_code_and_notebooks_mapped': len(expected), 'named_experiments': len(spec['experiments'])}


def documentation_links(root):
    names = ['README.md', 'docs/MENTOR_REVIEW.md', 'docs/PATH_MAP.md', 'src/README.md',
             'experiments/README.md', 'notebooks/README.md', 'paper/README.md', 'data/README.md',
             'tests/README.md', 'tools/README.md', 'archive/README.md', '.forge/README.md']
    names += ['experiments/' + e['directory'] + '/README.md' for e in layout(root)['experiments']]
    names += ['remediation/REPORT.md','paper/evidence_ledger.md','paper/plain_language_summary.md']
    links = 0
    for name in names:
        path = root / name
        for raw in re.findall(r'\[[^\]]*\]\(([^)]+)\)', path.read_text()):
            url = urlsplit(raw.strip('<>'))
            if url.scheme or not url.path:
                continue
            target = (path.parent / unquote(url.path)).resolve()
            require(target.is_relative_to(root.resolve()), f'Review link leaves project: {name}: {raw}')
            require(target.exists(), f'Broken review link: {name}: {raw}')
            links += 1
    return {'review_documents': len(names), 'local_links_checked': links}


def evidence_gate(root):
    base = root / layout(root)['archive_root']
    proc = subprocess.run([sys.executable, str(base / '.forge/check.py')], cwd=base,
                          capture_output=True, text=True, timeout=180)
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise ValueError(f'Preserved forge gate did not return JSON: {proc.stderr[-2000:]}')
    failures = [c for c in result['checks'] if not c['passed']]
    require(proc.returncode == 0 and result['status'] == 'passed', f'Preserved evidence gate failed: {failures}')
    return {'checks_passed': len(result['checks']), 'scope': result['scope']}


def remediation_gate(root):
    sys.path.insert(0,str(root/'tools'))
    import verify_remediation
    result=verify_remediation.verify(root)
    require(result['status']=='passed',f'Active remediation checks failed: {[c for c in result["checks"] if not c["passed"]]}')
    return {'checks_passed':len(result['checks'])}


CHECKS = (preserved_bytes, review_views, source_coverage, documentation_links, evidence_gate,remediation_gate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / '.forge/latest.json')
    args = parser.parse_args()
    results = []
    for check in CHECKS:
        try:
            results.append({'check': check.__name__, 'passed': True, **check(ROOT)})
        except Exception as exc:
            results.append({'check': check.__name__, 'passed': False, 'reason': str(exc)})
    result = {'status': 'passed' if all(c['passed'] for c in results) else 'failed',
              'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'checks': results,
              'scope': 'Byte preservation, report/source consistency and raw-output reanalysis. The new T4 competence screen failed and attack efficacy is unmeasured; different-model annotation review is not human adjudication. The full repeated T4 timing run failed from out-of-memory; shorter blocks are descriptive. No fresh model inference is performed by this gate.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
