"""Generate completion tables from audited raw evidence, never from saved flags."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_summary(root):
    from tools.verify_completion import verify
    result = verify(root)
    result['scope'] = ('Completed failed external competence screen; different-model blinded annotation review; '
                       'failed full T4 timing protocol with completed shorter blocks. No external attack efficacy or human adjudication established.')
    return result


def paper_block(summary):
    external, annotation, timing = (summary[k] for k in ('external', 'annotation', 'timing'))
    lines = ['<!-- COMPLETION_20261028 -->',
             '### 5.10. Completed prospective screen and blinded model review', '',
             'The October 28 T4 run used the frozen base Mamba-2.8B checkpoint, FP32, the fixed published-task prompts and unchanged competence criteria. It completed 72 new source-example generations and three instruction-only controls. The original runtime attempt failed before scoring a source example because the tokenizer joined trusted padding with the first untrusted word. A versioned adapter repair preserves the full prompt and its input token IDs; all 5,172 prepared records pass the exact AutoTokenizer serialization check. Both attempts and the repair are retained. No prompt, checkpoint, threshold or task was selected using these outcomes.', '',
             '| Task | Examples | Published parser correct | Strict correct | Strict valid | Gate |',
             '| --- | ---: | ---: | ---: | ---: | --- |']
    for task, label in [('sst2', 'SST-2'), ('sms_spam', 'SMS Spam'), ('mrpc', 'MRPC')]:
        row = external['gate'][task]
        lines.append(f'| {label} | 24 | {external["published_parser_correct"][task]}/24 | {row["strict_correct"]}/24 | {row["strict_valid"]}/24 | {"Pass" if row["passed"] else "Fail"} |')
    lines += ['', 'All three strict instruction-only gains are zero. The permissive published parser can recover labels from some paragraph answers; zero strict correctness therefore does not imply zero semantic answer content. The screen differs from the earlier development study in its source selection and fixed trailing-newline prompt, so the counts are not a controlled before/after comparison. The frozen strict-format gate failed, so the protocol stopped before attack-conditioned generation; before/after backbone hashes match. The 4,500 prepared attacks therefore provide no measured security efficacy result. This establishes a competence/format limitation of the tested checkpoint/prompt configuration. It does not settle whether a competent model with another preregistered configuration could benefit from routing.', '',
              f'A different-model blinded review used local Llama 3.2 3B at temperature zero with a pinned model digest. It received only the randomized V2 packet, one item per request, and completed {annotation["items"]} source items and {annotation["notes"]} notes in 32 requests. Relative to the retained annotations it disagreed on {annotation["note_disagreements"]}/{annotation["notes"]} note labels and {annotation["candidate_disagreements"]}/{annotation["items"]} supported candidates. These are disagreements, not an adjudicated error rate. Original labels remain unchanged. The abstract notes were generated/reviewed with Claude, while Llama had been used for an earlier separate synthetic pilot; this is bounded model/provider separation. No external human adjudication or returned human spot-check is available.', '',
              '### 5.11. Repeated T4 timing and a memory failure', '',
              f'The follow-up planned three warmups per mode at each prefix length, then 20 timed repetitions per mode. It completed {timing["warmups"]} warmups and {timing["timed"]}/{timing["planned_timed"]} planned measured paths before CUDA out-of-memory failure during the 2,048-token block. The 128- and 512-token blocks each completed all 20 repetitions per mode; only six measured paths completed at 2,048 tokens. The full protocol failed, and its terminal weight hash is unavailable. Four mode orders are shuffled within paired repetitions with a fixed seed. Batch size is one, precision FP32 and generation length eight tokens. CUDA synchronization brackets full-path timing, including prefix processing, candidate-state creation/copy, speculative advancement, selection and generation. Loading, garbage collection and allocator-cache clearing are excluded. The decisions are forced, so this measures cache-path cost without learned-router cost or attack efficacy.', '',
              '| Prefix tokens | Original median (s) | Forced accept median (s) | Forced reject median (s) | Filtering median (s) |',
              '| ---: | ---: | ---: | ---: | ---: |']
    stats = {(r['prefix_tokens'], r['mode']): r for r in timing['summary']}
    lengths = sorted({r['prefix_tokens'] for r in timing['summary']})
    for length in lengths:
        values = [stats[length, mode]['median_seconds'] for mode in
                  ['original', 'forced_accept', 'forced_reject', 'identical_decision_filtering']]
        lines.append('| ' + str(length) + ' | ' + ' | '.join(f'{v:.3f}' for v in values) + ' |')
    lines += ['', '| Prefix tokens | Accept/original median paired time ratio | Reject/filter median paired time ratio |',
              '| ---: | ---: | ---: |']
    comparisons = {(r['prefix_tokens'], r['candidate']): r for r in timing['paired_comparisons']}
    for length in lengths:
        cells = []
        for mode in ['forced_accept', 'forced_reject']:
            row = comparisons[length, mode]
            cells.append(f'{row["median_paired_time_ratio"]:.3f}')
        lines.append(f'| {length} | ' + ' | '.join(cells) + ' |')
    lines += ['', f'The tables are post-failure descriptive summaries of the completely observed shorter blocks. All {timing["equivalent_pairs"]} paired output-token comparisons in those blocks agree. Pre-timing cache isolation/continuation checks passed. The planned complete-run uncertainty analysis was not reached; no complete-run interval or terminal weight check is claimed. Raw rows retain peak allocated/reserved memory, and the derived summary includes p95 values. The out-of-memory error reported 13.83 GiB allocated by PyTorch on a 14.56 GiB T4. Earlier short feasibility successes therefore do not guarantee repeated two-bank execution at 2,048 tokens. These observations condition on one workload, runtime and T4.',
              '<!-- END_COMPLETION_20261028 -->']
    return '\n'.join(lines) + '\n'


def main():
    summary = make_summary(ROOT)
    (ROOT / 'remediation/completion_20261027/summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    block = paper_block(summary)
    (ROOT / 'paper/completion_results.md').write_text(block)
    path = ROOT / 'paper/manuscript.md'
    text = path.read_text()
    start, end = '<!-- COMPLETION_20261028 -->', '<!-- END_COMPLETION_20261028 -->'
    if start in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        text = before + block.rstrip() + after
    else:
        marker = '## 6. Discussion'
        if text.count(marker) != 1:
            raise ValueError('Expected one discussion section')
        text = text.replace(marker, block + '\n' + marker)
    path.write_text(text)


if __name__ == '__main__':
    main()
