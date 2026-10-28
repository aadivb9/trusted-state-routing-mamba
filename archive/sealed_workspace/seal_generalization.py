"""Freeze audited inputs and evaluation code before viewing any model outcomes."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'generalization_experiment'
FILES = [
    'build_generalization_data.py', 'evaluate_generalization.py',
    'clean_baseline_diagnostic.py',
    'generalization_experiment/protocol.json',
    'generalization_experiment/source_abstracts.json',
    'generalization_experiment/accepted_tasks.json',
    'generalization_experiment/semantic_audit.json',
    'generalization_experiment/dataset.json',
    *[f'generalization_experiment/audit_{i:02d}.json' for i in range(0, 32, 8)],
]


def seal():
    from build_generalization_data import assemble, require_unsealed
    require_unsealed()
    for name in ['routing.json', 'baseline_progress.json', 'execution_manifest.json']:
        if (OUT / name).exists():
            raise RuntimeError(f'Outcomes already exist: {name}; cannot seal retrospectively.')
    assemble()
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for name, expected in protocol['frozen_files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    result = {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'Audited data and evaluation code, frozen before model or router outcomes.',
        'files': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES},
    }
    with (OUT / 'seal.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print('Sealed', len(FILES), 'files; historical frozen inputs also verified.')


if __name__ == '__main__':
    seal()
