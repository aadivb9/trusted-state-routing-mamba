"""Update active review documents after the bounded final tests are audited."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def replace(name, old, new):
    path = ROOT / name
    text = path.read_text()
    if old not in text and new in text:
        return
    if text.count(old) != 1:
        raise ValueError('Expected one old passage: ' + name + ': ' + old[:70])
    path.write_text(text.replace(old, new))


def main():
    from tools.completion_summary import make_summary, main as build_tables
    summary = make_summary(ROOT)
    if not (all(g['strict_correct']==g['strict_valid']==0 for g in summary['external']['gate'].values())
            and summary['annotation']['note_disagreements']==47 and summary['annotation']['candidate_disagreements']==5
            and summary['timing']['timed']==166 and summary['timing']['equivalent_pairs']==80):
        raise ValueError('Raw results differ from this closeout narrative; review before publishing it')
    build_tables()
    completion = ROOT / 'remediation/completion_20261027'
    for name in ['remediation/finding_status.json', 'remediation/REPORT.md', 'remediation/narrative_review.json']:
        dest = completion / ('before_closeout_' + Path(name).name)
        if not dest.exists():
            shutil.copy2(ROOT / name, dest)
    status = json.loads((ROOT / 'remediation/finding_status.json').read_text())
    status.update(status='scoped_validation_completed_with_negative_results_and_limits', date='2026-10-28',
                  independent_annotation_review_completed=True, independent_annotation_review_kind='different_model_not_human',
                  human_annotation_review_completed=False, gpu_timing_repeated=False, gpu_timing_attempt_finished=True,
                  external_competence_screen_completed=True,
                  scope='All 12 findings retain explicit dispositions and residual scientific limits. The new T4 competence screen failed under the frozen gate; attack efficacy is unmeasured. Different-model review is not human adjudication.')
    updates = {
        3: ('New source-disjoint screen and different-model review completed',
            ['experiments/17_external_validation/results_v2.json', 'remediation/completion_20261027/model_review.json'],
            'The competence gate failed. Annotation disagreements remain unadjudicated; the data adaptation is not a full official reproduction.'),
        6: ('Prospective T4 competence screen completed and stopped under the frozen gate',
            ['experiments/17_external_validation/protocol.json', 'experiments/17_external_validation/results.json',
             'experiments/17_external_validation/results_v2.json', 'experiments/17_external_validation/runtime_amendment_v2.json'],
            'All three tasks have 0/24 strict-correct and 0/24 strict-format valid outputs. Attack-conditioned generation was not performed; external efficacy is unmeasured.'),
        9: ('Different-model blinded review completed; human adjudication unavailable',
            ['experiments/16_blinded_annotation/status.json', 'remediation/completion_20261027/model_review.json',
             'remediation/completion_20261027/model_review_protocol.json'],
            'The reviewer disagrees on 47/160 note labels and 5/32 supported candidates. Original labels remain unchanged. No external human review or returned author spot-check is available.'),
        12: ('Pinned CPU reference and T4 cache checks passed; full timing run failed',
             ['experiments/18_runtime_validation/cpu_results.json', 'experiments/18_runtime_validation/timing_results.json',
              'experiments/18_runtime_validation/timing_protocol.json'],
             'The repeated timing protocol failed at 2,048 tokens after 166/240 measurements. The shorter blocks have 20 repetitions per mode; the timing terminal weight hash is unavailable. Native conversion equivalence and the historical interrupted terminal hash remain unverified.')}
    for item in status['findings']:
        if item['finding'] in updates:
            item['status'], item['evidence'], item['residual_limitation'] = updates[item['finding']]
    (ROOT / 'remediation/finding_status.json').write_text(json.dumps(status, indent=2) + '\n')
    packet_path = ROOT / 'experiments/16_blinded_annotation/status.json'
    packet = json.loads(packet_path.read_text())
    review_path = 'remediation/completion_20261027/model_review.json'
    packet.update(status='different_model_review_complete_human_adjudication_unavailable',
                  independent_review_completed=True, human_spotcheck_completed=False,
                  review_kind='different_model_not_human', review_artifact=review_path,
                  review_artifact_sha256=hashlib.sha256((ROOT / review_path).read_bytes()).hexdigest(),
                  scope='Different-model review of the blinded V2 packet is complete. Disagreements are retained without changing original labels; no human adjudication is claimed.')
    packet_path.write_text(json.dumps(packet, indent=2) + '\n')

    replace('paper/manuscript.md',
            'Preparation of that packet does not constitute an independent review; a returned independent review and human spot-check remain pending.',
            'The completed follow-up supplies this packet to a different local model with a pinned digest, without labels or the private mapping. It is model-based review; no human adjudication is available. All disagreements remain in the record.')
    replace('paper/manuscript.md',
            'No new GPU execution occurred: local CUDA preflight failed.',
            'The subsequent T4 execution completed the 72-example screen after a versioned token-boundary runtime repair; all three tasks failed the unchanged gate (Section 5.10).')
    replace('paper/manuscript.md',
            'The replacement exporter removes these mechanical cues, but no returned independent review or human spot-check is available;',
            'The replacement exporter removes these mechanical cues; a different-model blinded review disagrees on 47/160 note labels and 5/32 candidates, without human adjudication;')
    replace('paper/manuscript.md',
            'The T4 trial uses two timing repetitions and reference PyTorch kernels; it does not support a general throughput claim.',
            'The historical T4 trial used two timing repetitions. The closing T4 follow-up planned 20 repetitions per path and three warmups at each length, with reference PyTorch kernels and forced decisions. It completed the 128- and 512-token blocks, then failed at 2,048 tokens after 166/240 measurements. Its terminal weight hash is unavailable. These results support no general throughput or memory-fit guarantee.')
    replace('paper/manuscript.md',
            'Neither this engineering repair nor the prepared prospective external protocol establishes useful transferable poisoning defense.',
            'The completed prospective screen fails clean-task competence, and the different-model annotation review leaves substantial disagreements. These outcomes do not establish useful transferable poisoning defense.')
    replace('paper/manuscript.md',
            'Cache mechanics pass prior T4 feasibility checks and a new pinned small-model CPU reference comparison, but neither establishes general speed or security advantages.',
            'Cache mechanics pass pinned small-model CPU and large-model T4 checks, while a closing T4 repetition study runs out of memory at 2,048 tokens; these measurements establish no general speed or security advantage.')
    replace('paper/manuscript.md',
            'The larger model establishes that GPU memory is not the immediate barrier for the tested inference workload. It does not remove the need for clean-task competence.',
            'The larger model fits the earlier short T4 feasibility workloads. The repeated follow-up reaches a memory limit at 2,048 tokens, so feasibility depends on the workload and runtime. Clean-task competence remains a separate requirement.')
    replace('README.md',
            'The independent annotation packet and new GPU efficacy study remain pending.',
            'The new T4 competence screen completed and failed all three tasks, so attack efficacy remains unmeasured. Different-model blinded annotation review completed with substantial disagreements, and the full repeated T4 timing run failed at 2,048 tokens after 166/240 measurements. See the [final validation record](remediation/completion_20261027/README.md).')
    replace('docs/MENTOR_REVIEW.md',
            'The [blinded reviewer packet](../experiments/16_blinded_annotation/reviewer_packet.zip) is prepared but not yet independently reviewed.',
            'The [blinded reviewer packet](../experiments/16_blinded_annotation/reviewer_packet.zip) received a different-model review: 47/160 note-label and 5/32 candidate disagreements remain unadjudicated. No human validation is claimed.')
    replace('paper/README.md',
            'The [audit response](../remediation/REPORT.md) distinguishes completed engineering/reporting corrections from pending external efficacy, independent annotation and GPU timing validation.',
            'The [audit response](../remediation/REPORT.md) records the completed failed T4 competence screen, different-model annotation disagreements and the failed full T4 timing run. External attack efficacy and human adjudication remain unestablished.')
    replace('paper/plain_language_summary.md',
            'The revised annotation packet removes answer-position and fixed-note-order cues, but independent review is still pending. A new published-data external protocol and source-disjoint records are prepared, but no new GPU clean gate or attack-conditioned generations have run. These unfinished validation requirements remain visible in the [finding-by-finding response](../remediation/REPORT.md).',
            'The final published-data competence test ran on a T4: all three tasks produced 0/24 strict-correct and 0/24 strict-format valid label outputs. The frozen protocol therefore stopped before attacking the model; this gives no external defense efficacy estimate. A different-model blinded review completed all 32 sources and 160 notes, disagreeing on 47 note labels and five candidate answers. Original labels remain unchanged and human adjudication is unavailable. The repeated T4 study completed 20 repetitions per mode at 128 and 512 tokens, then ran out of memory at 2,048 tokens after 166/240 planned measurements. See the [final validation record](../remediation/completion_20261027/README.md).')
    replace('paper/evidence_ledger.md',
            '| Reviewer presentation cues removed in new exporter | [Packet status](../experiments/16_blinded_annotation/status.json), [blinding code](../src/validation/blinded_review_v2.py) | No independent review or human spot-check returned |',
            '| Different-model blinded review completed | [Raw review](../remediation/completion_20261027/model_review.json), [packet status](../experiments/16_blinded_annotation/status.json) | 47/160 note and 5/32 candidate disagreements; no human adjudication |')
    replace('paper/evidence_ledger.md',
            '| New external efficacy remains unrun | [Preflight](../experiments/17_external_validation/preflight.json) | Local CUDA unavailable; no new gate or attack-conditioned inference |',
            '| New T4 competence screen completed and failed | [Raw generations](../experiments/17_external_validation/results_v2.json), [runtime repair](../experiments/17_external_validation/runtime_amendment_v2.json) | 0/24 strict-correct and valid outputs per task; no attack-conditioned efficacy estimate |\n| Full repeated T4 timing run failed | [Raw timing](../experiments/18_runtime_validation/timing_results.json), [protocol](../experiments/18_runtime_validation/timing_protocol.json) | 166/240 measurements; shorter blocks complete, 2,048-token block failed; forced decisions exclude router cost |')

    lines = ['# Audit response and final validation', '',
             'The twelve audit findings retain engineering dispositions and scientific limits. The new T4 competence screen is complete and failed its frozen gate; attack efficacy is unmeasured. Different-model blinded review is complete. The full repeated T4 timing run failed at 2,048 tokens after 166/240 planned measurements; its completed shorter blocks are reported descriptively. Human adjudication and native-checkpoint conversion parity are not established.', '',
             'The original audit, sealed results, failed GPU attempt and previous packages are preserved. See the [final validation record](completion_20261027/README.md) and [raw-derived summary](completion_20261027/summary.json).', '',
             '| Finding | Disposition | Evidence |', '| ---: | --- | --- |']
    for item in status['findings']:
        links = ', '.join(f'[{Path(name).name}](../{name})' for name in item['evidence'])
        lines.append(f'| {item["finding"]} | {item["status"]} | {links} |')
    lines += ['', 'Residual limits:', '']
    lines += [f'- Finding {item["finding"]}: {item["residual_limitation"]}' for item in status['findings']]
    lines += ['', 'Original labels, learned thresholds and historical outcomes are unchanged. The first new GPU attempt failed before scoring any development example; the versioned runtime repair leaves full-prompt strings and token IDs unchanged. No prompt or checkpoint sweep followed the new negative outcome. The annotation reviewer received only the randomized public packet; send only that packet to a future blinded reviewer, since the complete mentor package contains the private mapping.']
    (ROOT / 'remediation/REPORT.md').write_text('\n'.join(lines) + '\n')
    (completion / 'README.md').write_text('''# Final validation record - October 28, 2026

The two-bank mechanism works on the tested cache workloads. The research does not establish a useful transferable memory-poisoning defense or a predictive advantage unique to two banks.

- **External competence screen:** 72 new source-example generations plus three instruction-only controls on the pinned FP32 Mamba-2.8B/T4 setup. SST-2, SMS Spam and MRPC each score 0/24 strict-correct and 0/24 strict-format valid label outputs. The published permissive parser scores 14/24, 0/24 and 12/24 respectively; this is not zero semantic answer content. The frozen strict-format gate stopped the study before attack-conditioned generation. This is a completed failed competence screen; attack efficacy is unmeasured.
- **Different-model blinded review:** all 32 sources and 160 notes reviewed by pinned local Llama 3.2 3B. It disagrees on 47 note labels and five supported candidates. This is model/provider separation from the abstract-note generator, with the disclosed earlier separate Llama synthetic pilot. Original labels are unchanged. No external human adjudication or returned author spot-check is available.
- **Repeated T4 timing:** the full run failed from CUDA out-of-memory during the 2,048-token block after 36 warmups and 166/240 planned measurements. The 128- and 512-token blocks completed 20 repetitions per mode, with all 80 paired output-token comparisons matching. Cache checks passed before timing, but the failed timing run has no terminal weight hash. The shorter-block medians/p95 and paired ratios are descriptive; no complete-run interval is claimed. Fixed decisions exclude learned-router cost.

The first GPU attempt failed on token-boundary serialization before scoring a source example. Its [raw output](attempt1_20261028/results.json) is retained. The [versioned repair](../../experiments/17_external_validation/runtime_amendment_v2.json) preserves full-prompt text and token IDs; its actual AutoTokenizer audit covers all 5,172 prepared records. The separate lower-level tokenizer check is diagnostic only and is not the runtime compatibility evidence.

## Evidence and reproduction

- [Raw external screen](../../experiments/17_external_validation/results_v2.json)
- [Raw annotation review](model_review.json) and [frozen protocol](model_review_protocol.json)
- [Raw timing](../../experiments/18_runtime_validation/timing_results.json) and [frozen protocol](../../experiments/18_runtime_validation/timing_protocol.json)
- [Recomputed summary](summary.json)
- [Self-contained reproduction notebook](../../notebooks/16_final_validation_reproduction.ipynb)
- [Current manuscript](../../paper/manuscript.md) and [PDF](../../paper/manuscript.pdf)

From the workspace root, `python3 tools/verify_completion.py` independently recomputes scores, disagreements and timing summaries from saved rows. It performs no new inference. `python3 tools/verify.py` also checks sealed evidence, current reports and output freshness. The package builder repeats the gate and CPU controls in an empty extraction directory.

For fresh GPU reproduction, use the notebook with PyTorch 2.11.0 on a T4. Its payload is verified, but that exported notebook has not itself been run as an independent replication. All output paths refuse overwrite. Keep any reproduction in a fresh directory and retain failures; do not tune prompts, checkpoints, labels or thresholds after seeing outcomes.

## Interpretation limits

The earlier 28-to-80 two-choice synthetic gain remains conditional on its narrow schema. An unfitted rule matches the cascade decisions, and matched filtering matches the recorded controller outputs. The new competence failure supplies no attack-success estimate. Model-review disagreements supply no adjudicated annotation accuracy. Human validation, native instruction-checkpoint conversion parity, full official RoBench/BIPIA utility evaluations and the interrupted historical segment's missing terminal hash remain outside the established evidence. These are explicit limits of the paper, not successful tests.
''')
    (ROOT / 'experiments/17_external_validation/README.md').write_text('''# Completed external competence screen

The repaired FP32 Mamba-2.8B T4 run completed the frozen 72-example screen. All three tasks have 0/24 strict-correct and 0/24 strict-format valid outputs. The gate stopped before attack-conditioned generation; external defense efficacy is unmeasured.

- [Raw repaired run](results_v2.json)
- [Preserved first runtime failure](results.json)
- [Frozen study protocol](protocol.json) and [versioned runtime repair](runtime_amendment_v2.json)
- [Reproduction notebook](../../notebooks/16_final_validation_reproduction.ipynb)
- [Final validation record](../../remediation/completion_20261027/README.md)

The original notebook/source version and its runtime amendment are retained. The repaired run keeps the same checkpoint, prepared source records, full-prompt strings/token IDs, parsers, decoding and gate. No prompt/checkpoint sweep followed the result. Published attack constructors and the hash-selected subset remain adaptations rather than a full official benchmark reproduction.
''')
    (ROOT / 'experiments/18_runtime_validation/README.md').write_text('''# Pinned reference, cache and timing validation

The small-model CPU reference check and large-model T4 cache checks passed. The full repeated T4 run failed from CUDA out-of-memory at 2,048 tokens after 36 warmups and 166/240 planned measurements. The 128- and 512-token blocks completed 20 repetitions per mode; all 80 output-token pair checks in those blocks agree. The failed timing run has no terminal weight hash.

- [CPU reference result](cpu_results.json)
- [Raw T4 timing](timing_results.json), [frozen timing protocol](timing_protocol.json) and [driver](run_timing.py)
- [Final validation record](../../remediation/completion_20261027/README.md)

These timings use forced decisions on fixed synthetic text and exclude learned-router cost. Native instruction-checkpoint conversion parity is not established. The original prospective plan remains preserved in protocol.json.
''')


if __name__ == '__main__':
    main()
