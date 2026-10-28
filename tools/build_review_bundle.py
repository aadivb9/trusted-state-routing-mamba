"""Build a portable local mentor package, then test it in an empty extraction."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'dist'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def run(root, command):
    proc = subprocess.run([sys.executable, *command], cwd=root, capture_output=True, text=True, timeout=180)
    if proc.returncode:
        raise RuntimeError(f'{command} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}')
    return proc


def selected_files():
    # followlinks materializes the named review views for common ZIP extractors.
    for folder, directories, names in os.walk(ROOT, followlinks=True):
        base = Path(folder)
        relative = base.relative_to(ROOT)
        directories[:] = sorted(d for d in directories if d not in ('__pycache__', '.git')
                                 and not (relative == Path('.') and d in ('dist', '.codex', '.playwright-mcp')))
        for name in sorted(names):
            if name == '.DS_Store' or name.endswith('.pyc'):
                continue
            path = base / name
            rel = str(path.relative_to(ROOT))
            if rel.startswith('.forge/') and path.suffix == '.json':
                continue  # Fresh root-level check output is generated after extraction.
            if rel == 'remediation/package_build.log':
                continue  # The caller may be redirecting this build's live output here.
            if not path.resolve().is_relative_to(ROOT.resolve()):
                raise RuntimeError(f'Link outside workspace: {rel}')
            yield rel, path


def main():
    OUT.mkdir(exist_ok=True)
    (OUT/'verification.json').write_text(json.dumps({'status':'building','scope':'A new build is in progress; prior success is superseded.'},indent=2)+'\n')
    run(ROOT, ['tools/verify.py'])
    spec = json.loads((ROOT / 'docs/layout.json').read_text())
    original = ROOT / spec['archive_root']
    code = ['MENTOR CODE REVIEW\n\nStart with docs/MENTOR_REVIEW.md in the ZIP.\n'
            'Files below are byte-exact first-party review sources; use tools/run.py for execution.\n']
    seen = set()
    for name, item in spec['views'].items():
        if item['kind'] == 'file' and name.endswith('.py') and item['original'] not in seen:
            seen.add(item['original'])
            code.append('\n' + '=' * 78 + f'\n{name}\nPreserved source: {item["original"]}\n' + '=' * 78 + '\n')
            code.append((original / item['original']).read_text())
    for name in spec.get('active_versions',{}):
        if name.endswith('.py'):
            code.append('\n'+'='*78+f'\n{name}\n{spec["active_versions"][name]["purpose"]}\n'+'='*78+'\n')
            code.append((ROOT/name).read_text())
    for item in spec['experiments']:
        protocol = ROOT / 'experiments' / item['directory'] / 'protocol.json'
        if protocol.exists():
            code.append('\n' + '=' * 78 + f'\n{protocol.relative_to(ROOT)}\n' + '=' * 78 + '\n')
            code.append(protocol.read_text())
    text_path = OUT / 'mentor_code.txt'
    text_path.write_text('\n'.join(code))
    archive = OUT / 'mentor_review.zip'
    manifest = {}
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, path in selected_files():
            raw = path.read_bytes()
            manifest[name] = sha(raw)
            z.writestr(name, raw)
        manifest['mentor_code.txt'] = sha(text_path.read_bytes())
        z.write(text_path, 'mentor_code.txt')
        z.writestr('review_manifest.json', json.dumps({'files': manifest, 'scope': 'Portable mentor-review files; no model weights or public release.'}, indent=2) + '\n')
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist()) != len(set(z.namelist())):
            raise RuntimeError('Duplicate ZIP paths')
        for name, digest in manifest.items():
            if sha(z.read(name)) != digest:
                raise RuntimeError(f'ZIP hash mismatch: {name}')
            source=text_path if name=='mentor_code.txt' else ROOT/name
            if sha(source.read_bytes())!=digest:
                raise RuntimeError(f'ZIP source changed while building: {name}')
    with tempfile.TemporaryDirectory(prefix='mentor-review-package-') as tmp:
        with zipfile.ZipFile(archive) as z:
            z.extractall(tmp)
        check = json.loads(run(Path(tmp), ['tools/verify.py']).stdout)
        tests = run(Path(tmp), ['tools/run.py', 'tests'])
        new_tests=run(Path(tmp),['tools/run.py','remediation-tests'])
        (OUT / 'isolated_cpu_tests.log').write_text(tests.stdout + tests.stderr)
        (OUT / 'isolated_remediation_tests.log').write_text(new_tests.stdout+new_tests.stderr)
    result = {'status': 'passed', 'archive': str(archive.relative_to(ROOT)),
              'files_verified': len(manifest), 'archive_bytes': archive.stat().st_size,
              'archive_sha256': sha(archive.read_bytes()), 'review_text_sha256': sha(text_path.read_bytes()),
              'current_source_parity':'passed','isolated_project_gate': check['status'], 'isolated_cpu_tests': 'passed','isolated_remediation_tests':'passed',
              'scope': 'Local review bundle, materialized views, byte parity, and isolated saved-output/CPU checks; no fresh GPU inference.'}
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        OUT.mkdir(exist_ok=True)
        (OUT/'verification.json').write_text(json.dumps({'status':'failed','error':str(exc)[-3000:],'scope':'Do not treat the partially built ZIP as a verified package.'},indent=2)+'\n')
        raise
