"""Replay the immutable text-router benchmark under its original source hashes.

Historical names exist only inside the archived reproduction. Active project
names are descriptive; the archive retains the exact evaluated bytes.
"""
import importlib.util
from pathlib import Path
import tempfile
import zipfile

from build_text_router_notebook import NAMES

_work=Path(tempfile.mkdtemp(prefix='text_router_reproduction_'))
with zipfile.ZipFile(Path(__file__).with_name('text_router_benchmark_archive.zip')) as _archive:
    for _name in NAMES:
        _target=_work/_name
        _target.parent.mkdir(parents=True,exist_ok=True)
        _target.write_bytes(_archive.read(_name))
_spec=importlib.util.spec_from_file_location('_sealed_text_router_evaluation',_work/'fresh_robustness_eval.py')
_module=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
globals().update({name:value for name,value in vars(_module).items() if not name.startswith('_')})

if __name__=='__main__':
    print('Historical reproduction directory:',_work)
    _module.evaluate()
