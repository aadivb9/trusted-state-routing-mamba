"""Run named maintenance commands in the preserved experiment environment."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / 'archive/sealed_workspace'
COMMANDS = {
    'tests': ['-m', 'unittest', 'test_adaptive_memory_router', 'test_clean_baseline_diagnostic',
              'test_generalization', 'test_generalization_boundary_fix', 'test_text_router_benchmark', '-v'],
    'forge-tests': ['-m', 'unittest', 'discover', '-s', '.forge', '-p', 'test_*.py', '-v'],
    'audit-hispa': ['revision_experiments/hispa_stress/audit.py'],
    'audit-instruction-model': ['revision_experiments/competence/audit.py'],
    'audit-router-transfer': ['revision_experiments/router_transfer/audit.py'],
    'audit-earlier-results': ['research_closeout/audit_existing.py'],
    'audit-closing-baselines': ['research_closeout/audit_final.py'],
}
ACTIVE_COMMANDS={
    'build-paper':['paper/build.py'],
    'build-evidence':['tools/build_review_bundle.py'],
    'remediation-tests':['-m','unittest','discover','-s','tests','-p','test_remediation*.py','-v'],
    'audit-remediation':['experiments/15_router_remediation/analyze.py'],
    'embedding-replay':['experiments/15_router_remediation/reconstruct_embeddings.py'],
    'external-preflight':['experiments/17_external_validation/run.py'],
    'audit-followup':['tools/followup_summary.py'],
    'followup-evidence-tests':['-m','unittest','discover','-s','tests','-p','test_followup_evidence.py','-v'],
    'followup-serialization-tests':['-m','unittest','discover','-s','tests','-p','test_followup_serialization.py','-v'],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=sorted(COMMANDS.keys()|ACTIVE_COMMANDS.keys()), nargs='?')
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 0
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT)+os.pathsep+str(WORKSPACE) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    active=args.command in ACTIVE_COMMANDS
    command=ACTIVE_COMMANDS[args.command] if active else COMMANDS[args.command]
    proc = subprocess.run([sys.executable, *command], cwd=ROOT if active else WORKSPACE, env=env)
    return proc.returncode


if __name__ == '__main__':
    sys.exit(main())
