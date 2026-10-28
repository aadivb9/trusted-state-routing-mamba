"""Render the sealed evaluation's measurements without changing its protocol."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'text_router_benchmark'


def percentage(value):
    return f'{100 * value:.1f}%'


def accuracy_cell(value):
    lower, upper = value['wilson_95']
    return f"{value['correct']}/{value['n']} ({percentage(value['accuracy'])}; 95% interval {percentage(lower)}–{percentage(upper)})"


def build():
    results = json.loads((OUTPUT / 'results.json').read_text())
    accounting = json.loads((OUTPUT / 'generation_accounting.json').read_text())
    seal = json.loads((OUTPUT / 'seal.json').read_text())
    router = results['router_metrics']
    equivalence = results['filtering_equivalence']
    original = results['summary']['original']
    dual = results['summary']['dual_state']
    difference = results['paired_differences']['malicious']['correct']
    lines = [
        '# Text-only router: independent attack benchmark', '',
        '**The independent test does not reproduce the earlier strong defense result.**'
        if not results['all_primary_criteria_passed'] else
        '**The independent test passes all predeclared primary criteria.**', '',
        'The backbone, fitted text router, threshold, cloze prompt, and scoring rules were frozen before evaluation. '
        'This set contains 100 new source groups, each with one safe and one malicious note: 200 new records, '
        'plus clean versions of the 100 complete source contexts. No record was used for training or tuning.', '',
        '## Accuracy', '',
        'Candidate ranking compares the probabilities of two complete answer continuations. '
        'Free generation uses greedy 16-token continuations and a predeclared value-at-start parser. '
        'Unparseable free answers count as wrong, including explanations that place the correct value later in the text. '
        'Free-generation accuracy therefore measures compliance with the requested answer format as well as retrieval. '
        'These are different measurements.', '',
        '| Measurement | Original Mamba-130m | Dual state |',
        '| --- | ---: | ---: |',
    ]
    for condition, title in [('clean', 'Clean'), ('safe', 'Safe note'), ('malicious', 'Malicious note')]:
        for metric, metric_title in [('candidate', 'candidate ranking'), ('free_generation', 'free generation')]:
            lines.append(f'| {title}, {metric_title} | {accuracy_cell(original[condition][metric])} | {accuracy_cell(dual[condition][metric])} |')
    lower, upper = difference['bootstrap_95']
    lines += [
        '',
        f"The attacked candidate-accuracy difference is **{100 * difference['difference']:+.1f} percentage points**, "
        f'with a paired source-bootstrap 95% interval of **{100 * lower:+.1f} to {100 * upper:+.1f} points**. '
        'The bootstrap uses 5,000 resamples with the saved seed. Clean/safe/attacked observations from the same source are not treated as unrelated cases.', '',
        'When every observed paired difference is zero, this empirical bootstrap interval collapses to zero. '
        'That does not establish population equivalence or rule out differences on unseen cases.', '',
        '| Free-generation invalid answers | Original | Dual state |',
        '| --- | ---: | ---: |',
    ]
    for condition in ['clean', 'safe', 'malicious']:
        lines.append(f"| {condition} | {original[condition]['invalid_free_answers']}/100 | {dual[condition]['invalid_free_answers']}/100 |")
    lines += ['', 'Twenty safe notes supply a fact deliberately omitted from their trusted context:', '',
              '| Required safe evidence | Original | Dual state |', '| --- | ---: | ---: |']
    for metric in ['candidate', 'free_generation']:
        lines.append(f"| {metric} | {accuracy_cell(original['useful_safe'][metric])} | {accuracy_cell(dual['useful_safe'][metric])} |")
    lines += [
        '', '## Router failure analysis', '',
        f"The frozen router catches **{router['tp']}/100 malicious notes** and misses **{router['fn']}/100**. "
        f"It rejects **{router['fp']}/100 safe notes**. Overall classification accuracy is {percentage(router['accuracy'])}; "
        f"attack recall is {percentage(router['recall'])}.", '',
        'These attacks are confident false research notes, metadata claims, quoted false results, and false summaries wrapped in benign prose. '
        'Safe notes include quoted instructions to reject and forceful reproducibility advice. '
        'The router sees only note text, not the trusted facts or question. Consequently it cannot directly check whether a note contradicts the source. '
        'The failure is consistent with reliance on the wording of the earlier synthetic examples. '
        'Generator, wording, and source-format changes occur together, so this run does not isolate a single cause.', '',
        '| Note family (20 each) | Original ranked | Dual ranked | Original free | Dual free | Router rejections |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for family, metrics in sorted(results['by_family'].items()):
        cells = [percentage(metrics[variant][metric]['accuracy'])
                 for metric in ['candidate', 'free_generation'] for variant in ['original', 'dual_state']]
        lines.append('| ' + family + ' | ' + ' | '.join(cells) + f" | {metrics['routing_rejections']}/20 |")
    lines += [
        '', '## What does memory separation add?', '',
        'A diagnostic uses the same frozen backbone and the same router decisions, removing rejected notes before inference. '
        'It is an equivalence check, not another trained model.', '',
        f"Candidate answers match dual state on **{equivalence['candidate_matches']}/{equivalence['n']}** records; "
        f"parsed free answers match on **{equivalence['free_prediction_matches']}/{equivalence['n']}**, "
        f"and complete generated texts match on **{equivalence['full_text_matches']}/{equivalence['n']}**. "
        f"Maximum candidate log-likelihood difference is **{equivalence['max_log_likelihood_error']:.8f}**.", '',
        'For this single-span experiment, rejected candidate memory contributes nothing to the subsequent question. '
        'The measured equivalence therefore provides no evidence of an accuracy advantage from storing two banks over filtering with the same router. '
        'It verifies the restore mechanism but does not establish a new learned Mamba architecture.', '',
        '## Speed and memory', '',
        'Twenty fixed records, three repeats per variant, 16 generated tokens per call, warm-up, and alternating order. '
        'Both variants explicitly request last-token logits during generation. CUDA synchronization brackets generation and the complete path. '
        'Full time includes tokenization, routing, extraction, cloning, external-note processing, prefill, and generation; candidate scoring is separate. '
        'The original performs context prefill inside `.generate()`, while dual state performs most of it before that call. '
        'Generation-only rates therefore cover different amounts of work; use the full-path rates for the comparison. '
        'These controlled settings supersede the earlier small speed pilot for interpreting this experiment.', '',
        '| Measurement | Original | Dual state |', '| --- | ---: | ---: |',
    ]
    for key, title, unit in [
        ('end_to_end_tokens_per_second', 'Full generation tokens/sec', ''),
        ('generation_only_tokens_per_second', 'Generation-only tokens/sec (excludes preparation)', ''),
        ('max_peak_allocated_bytes', 'Maximum peak CUDA allocated memory', 'MiB'),
        ('max_extra_peak_bytes', 'Maximum extra peak above pre-call allocation', 'MiB'),
        ('two_bank_capacity_bytes', 'Two-bank tensor capacity (derived for dual state)', 'MiB'),
    ]:
        values = [results['summary'][variant]['timing'][key] for variant in ['original', 'dual_state']]
        if unit:
            values = [value / 1024**2 for value in values]
        original_value = 'Not separately measured' if key == 'two_bank_capacity_bytes' else f'{values[0]:.2f} {unit}'
        lines.append(f'| {title} | {original_value} | {values[1]:.2f} {unit} |')
    lines += [
        '', f"Dual/original end-to-end generation throughput ratio: **{results['speed_ratio']:.3f}**. "
        'These are batch-one, implementation-specific measurements. Peak memory excludes CPU snapshots and reserved-but-unallocated CUDA memory. '
        'The two-bank figure is tensor capacity; the unused bank can be freed before generation. The original also has a generation cache, '
        'which the dual-specific capacity field does not separately expose.', '',
        '| Candidate scoring examples/sec (separate workload) | Original | Dual state |', '| --- | ---: | ---: |',
    ]
    for condition in ['clean', 'safe', 'malicious']:
        lines.append(f"| {condition} | {original[condition]['candidate_scoring_examples_per_second']:.2f} | {dual[condition]['candidate_scoring_examples_per_second']:.2f} |")
    lines += ['', '## Predeclared success criteria', '', '| Criterion | Passed |', '| --- | --- |']
    for criterion, passed in results['success_criteria'].items():
        lines.append(f'| {criterion} | {passed} |')
    lines += [
        '', f"All primary criteria passed: **{results['all_primary_criteria_passed']}**.",
        'The data, threshold, prompt, and scoring were not changed after seeing these outcomes.', '',
        '## Provenance and verification', '',
        '- New source names, paraphrased contexts, and new notes; no exact span or source-group overlap with the original 1,000 records.',
        '- Claude Code generated notes and performed separate label-blinded semantic audits. The same provider performed both; this is not independent human annotation.',
        '- One safe note contained an unsupported method name. It was corrected against the controlled source and reviewed before sealing; raw calls and the correction remain saved.',
        '- Claude CLI does not expose temperature. No 0.7 claim applies to this new dataset; the earlier Ollama dataset used 0.7.',
        f"- {accounting['calls']} generation/audit calls; CLI-reported list-price equivalent **${accounting['reported_list_price_usd']:.4f}**. "
        'Existing Claude subscription authentication was used; this is not a verified incremental charge. No OpenAI API calls.',
        '- The four CPU control tests pass. GPU state checks reach all 48 convolution/SSM tensors and verify independent storage, restoration, and generation handoff.',
        f"- Backbone SHA-256 is unchanged: `{results['backbone_after_sha256']}`.",
        f"- Dataset SHA-256: `{seal['dataset_sha256']}`.",
        f"- Evaluated source SHA-256: `{results['manifest']['source_sha256']}`.",
        f"- GPU: {results['manifest']['gpu']}; runtime: `{json.dumps(results['manifest']['versions'], sort_keys=True)}`.", '',
        'Run `../text_router_benchmark.ipynb` top to bottom in Colab T4. It embeds the sealed data and exact frozen implementation, '
        'creates a new output folder, and prints both accuracy measures, timing, routing errors, and equivalence results. '
        'The GPU script was executed; notebook source and embedded files were checked locally.', '',
        'The next design question is whether routing can use trusted facts to detect contradictions and whether the two-state mechanism offers a capability beyond filtering. '
        'This test must remain a reported evaluation set rather than become unacknowledged training data. '
        'The current result is limited to controlled synthetic cloze retrieval, known span boundaries, and these fixed attacks; it is not optimized HiSPA evaluation.',
    ]
    (OUTPUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('Wrote', OUTPUT / 'REPORT.md')


if __name__ == '__main__':
    build()
