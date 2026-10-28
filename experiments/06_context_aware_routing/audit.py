"""Reuse the label-blinded semantic audit for the new evaluation notes."""
from pathlib import Path
import audit_text_router_benchmark as audit

if __name__=='__main__':
    audit.OUTPUT=Path(__file__).resolve().parent/'adaptive_router_experiment'/'independent_benchmark'
    audit.run()
