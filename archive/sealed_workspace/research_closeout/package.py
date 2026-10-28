"""Create a local evidence package and verify every archived byte hash."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent.parent


def main():
    final=json.loads((ROOT/'research_closeout/run_03/verification.json').read_text())
    assert final['status']=='passed'
    names=set()
    for folder in ['paper','research_closeout','external_benchmark','mamba_28b_feasibility','final_hispa_audit']:
        names.update(str(p.relative_to(ROOT)) for p in (ROOT/folder).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.zip']
                     and p.name not in ['evidence_manifest.json','package_verification.json','results.partial.json'])
    for name in ['dual_state_mamba.py','adaptive_memory_router.py','cloze_dual_state_eval.py',
                 'adaptive_router_experiment/router.json','adaptive_router_experiment/results.json',
                 'adaptive_router_experiment/independent_benchmark/dataset.json',
                 'text_router_benchmark/results.json','text_router_benchmark/dataset.json',
                 'text_router_benchmark/filtering_equivalence.json','generalization_experiment/results.json',
                 'generalization_experiment/dataset.json','FINAL_RESEARCH_SUMMARY.md',
                 'mamba_28b_t4_feasibility.ipynb','external_benchmark_competence.ipynb']:
        names.add(name)
    names.update(['research_closeout/run_01/results.partial.json','research_closeout/run_02/results.partial.json'])
    # Include the original archives required by the external preparation audit.
    names.update(str(p.relative_to(ROOT)) for p in (ROOT/'external_benchmark/raw').glob('*.zip'))
    manifest={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sorted(names)}
    manifest_path=ROOT/'research_closeout/evidence_manifest.json'
    manifest_path.write_text(json.dumps(dict(scope='Local closing evidence package; not a public dataset release or complete archive of every historical development file',files=manifest),indent=2)+'\n')
    archive=ROOT/'research_closeout_evidence.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as stream:
        for name in sorted(names):stream.write(ROOT/name,name)
        stream.write(manifest_path,str(manifest_path.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as stream:
        for name,expected in manifest.items():
            assert hashlib.sha256(stream.read(name)).hexdigest()==expected,name
    verification=dict(status='passed',files_verified=len(manifest),archive=archive.name,
                      bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (ROOT/'research_closeout/package_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':
    main()
