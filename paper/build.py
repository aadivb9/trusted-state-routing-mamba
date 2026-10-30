"""Render the active revision while reading immutable historical evidence."""
import hashlib,importlib.metadata,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'paper';BASE=ROOT/'archive/sealed_workspace'


def main():
    source=BASE/'paper/build.py'
    spec=importlib.util.spec_from_file_location('preserved_renderer',source)
    renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
    renderer.ROOT=BASE;renderer.OUT=OUT;renderer.FIG=OUT/'figures'
    renderer.finalize_table();renderer.figures();renderer.render()
    inputs=['paper/manuscript.md','paper/build.py','paper/references.bib','paper/build_requirements.txt',
            'archive/sealed_workspace/paper/build.py','remediation/update_paper.py',
            'experiments/15_router_remediation/results.json','experiments/15_router_remediation/embedding_reconstruction.json',
            'experiments/18_runtime_validation/cpu_results.json','experiments/17_external_validation/preflight.json',
            'experiments/16_blinded_annotation/status.json']
    inputs += ['experiments/17_external_validation/results.json','experiments/17_external_validation/results_v2.json',
               'experiments/17_external_validation/runtime_amendment_v2.json',
               'experiments/18_runtime_validation/timing_results.json','experiments/18_runtime_validation/timing_protocol.json',
               'remediation/completion_20261027/model_review.json','remediation/completion_20261027/summary.json',
               'paper/completion_results.md','tools/completion_summary.py','tools/verify_completion.py',
               'remediation/finish_closeout.py','remediation/completion_20261027/model_review_protocol.json',
               'remediation/completion_20261027/serialization_auto_tokenizer_audit.json']
    inputs += ['archive/sealed_workspace/'+n for n in ['adaptive_router_experiment/results.json','text_router_benchmark/results.json','generalization_experiment/results.json','mamba_28b_feasibility/fp32_run_02/results.json','research_closeout/run_03/results.json','research_closeout/run_03/verification.json']]
    inputs += ['tools/followup_summary.py','tools/followup_report.py','paper/followup_results.md',
               'experiments/19_embedding_followup/protocol.json','experiments/19_embedding_followup/results.json',
               'experiments/19_embedding_followup/selection.json','experiments/19_embedding_followup/calibration_replay.json',
               'experiments/20_sst2_followup/protocol.json','experiments/20_sst2_followup/final_protocol.json',
               'experiments/20_sst2_followup/screen_results.json','experiments/20_sst2_followup/final_results.json',
               'experiments/20_sst2_followup/continuation_amendment.json','experiments/20_sst2_followup/continuation_overlap.json',
               'experiments/20_sst2_followup/mask_replay.json',
               'remediation/followup_20261029/summary.json','remediation/followup_20261029/extra_normalization_probe.json']
    outputs=['paper/manuscript.pdf','paper/manuscript.html','paper/figure_data.csv','paper/final_baseline_table.md']
    outputs += [str(p.relative_to(ROOT)) for p in sorted((OUT/'figures').iterdir()) if p.suffix in ('.pdf','.png')]
    hashes=lambda names:{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    result={'inputs':hashes(inputs),'outputs':hashes(outputs),
            'environment':{n:importlib.metadata.version(n) for n in ('markdown','matplotlib','numpy','reportlab')},
            'scope':'Active paper revision and immutable evidence; build provenance does not replace semantic or visual review.'}
    (OUT/'build_verification.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
