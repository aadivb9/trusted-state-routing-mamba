"""Run the explicitly data-independent public implementation controls."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'archive/sealed_workspace')]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    loader, suite = unittest.TestLoader(), unittest.TestSuite()
    router = load('public_router_regressions', 'tests/test_remediation.py')
    for cls in [router.RouterRegressions, router.CalibrationRegressions]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    for i, file in enumerate(['test_remediation_serialization.py', 'test_remediation_serialization_v2.py',
                              'test_remediation_timing.py', 'test_public_extractor.py','test_followup_serialization.py']):
        suite.addTests(loader.loadTestsFromModule(load('public_checks_' + str(i), 'tests/' + file)))
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
