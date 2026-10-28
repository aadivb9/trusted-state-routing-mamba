"""Read-only verification through the GitHub CLI; no external content execution."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'archive/sealed_workspace'
COMMIT = '898d7f7b78ba8f934f70340d12e9c1183bb623e5'
REPOSITORY = 'repos/TortueSagace/hispa'


def api(path, raw=False):
    args = ['gh', 'api', f'{REPOSITORY}/{path}']
    if raw:
        args += ['-H', 'Accept: application/vnd.github.raw+json']
    return subprocess.check_output(args, timeout=30)


def main():
    tree = json.loads(api(f'git/trees/{COMMIT}?recursive=1'))
    assert not tree.get('truncated', False)
    blobs = {r['path']: r['sha'] for r in tree['tree'] if r['type'] == 'blob'}
    items = json.loads((BASE/'robench25.json').read_text())['items']
    rows = []
    for r in items:
        raw = r['abstract'].encode()
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        path = f'data/NeurIPS2025_{r["id"]}.txt'
        rows.append(dict(id=r['id'], path=path, local_git_blob=blob,
                         upstream_git_blob=blobs[path], matches=blob == blobs[path]))
    labels = json.loads(api(f'contents/data/__labels.json?ref={COMMIT}', raw=True))
    titles, questions = 0, 0
    for r in items:
        original = labels[f'NeurIPS_{r["id"]}']
        titles += r['title'] == original['title']
        questions += sum(original.get(q['question']) == q['answer'] for q in r['questions'])
    provenance = json.loads((BASE/'final_hispa_audit/source_provenance.json').read_text())
    triggers = []
    for path, expected in provenance['files'].items():
        digest = hashlib.sha256(api(f'contents/{path}?ref={COMMIT}', raw=True)).hexdigest()
        triggers.append(dict(path=path, expected_sha256=expected,
                             upstream_sha256=digest, matches=digest == expected))
    report = dict(status='passed' if all(r['matches'] for r in rows+triggers)
                  and titles == 120 and questions == 240 else 'failed',
                  repository='https://github.com/TortueSagace/hispa', commit=COMMIT,
                  abstracts_compared=len(rows), abstract_exact_matches=sum(r['matches'] for r in rows),
                  upstream_title_matches=titles, upstream_question_label_matches=questions,
                  trigger_source_checks=triggers, abstracts=rows,
                  local_input_sha256={p: hashlib.sha256((BASE/p).read_bytes()).hexdigest()
                      for p in ('robench25.json','final_hispa_audit/source_provenance.json')},
                  scope='Direct GitHub API verification against the cited commit. Exact upstream parity does not authenticate the original publications or exclude pretraining exposure.')
    (HERE/'upstream_provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('abstracts','local_input_sha256')}, indent=2))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
