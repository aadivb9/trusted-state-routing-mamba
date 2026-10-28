"""Render measured transfer results, including stopped or GPU-pending runs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'generalization_experiment'


def rate(metric):
    return f"{metric['correct']}/{metric['n']} ({metric['accuracy']:.1%})"


def render():
    records = json.loads((OUT / 'dataset.json').read_text())['records']
    audit = json.loads((OUT / 'semantic_audit.json').read_text())
    result_path = OUT / 'results.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    interrupted_gate_path = OUT / 'interrupted_run/clean_gate.json'
    interrupted_gate = (json.loads(interrupted_gate_path.read_text())
                        if not result and interrupted_gate_path.exists() else None)
    pending_status = ('GPU run interrupted; repaired run not yet verified'
                      if interrupted_gate else 'GPU evaluation pending')
    lines = [
        '# Frozen-router generalization pilot', '',
        f"Status: **{result['status'] if result else pending_status}**.", '',
        f"This pilot uses {len(records)} previously unevaluated paper abstracts from the supplied RoBench file, "
        'with one cloze task and five synthetic notes per paper. The existing adaptive router and Mamba weights remain frozen. '
        'The sources have not been independently authenticated against published papers.', '',
        f"All {audit['n']} tasks received a separate semantic audit; unresolved issues: {len(audit['unresolved'])}. "
        'The same provider generated and reviewed the text. Note labels were hidden, but note order and the supported '
        'candidate position were fixed. This is not independent human annotation.', '',
        '## Conditions', '',
        '| Condition | Trusted context and incoming notes |',
        '| --- | --- |',
        '| Clean | Full abstract, no notes |',
        '| Safe addition | Supporting sentence removed; support, benign, and rejected-quotation notes |',
        '| Early attack | Full abstract; malicious, support, benign notes |',
        '| Late attack | Full abstract; support, benign, malicious notes |',
        '| Mixed note | Supporting sentence removed; benign, mixed truthful/false, rejected-quotation notes |', '',
        'The router compares each note with the immutable trusted context. Accepted notes update model memory '
        'but do not update the router reference. A rejected-quotation note can itself repeat the correct answer; '
        'the mixed-note condition therefore does not isolate recovery from the mixed note alone.', '',
    ]
    if result and 'summary' in result:
        original = result['summary']['original']['attack_late']['candidate']
        dual = result['summary']['dual_state']['attack_late']['candidate']
        difference = result['paired_differences']['attack_late']
        finding = (f"Late-attack ranked accuracy: **{rate(original)} original versus {rate(dual)} dual state**. "
                   f"Paired gain: {difference['difference']:+.1%}; 95% paper-bootstrap interval "
                   f"{difference['bootstrap_95'][0]:+.1%} to {difference['bootstrap_95'][1]:+.1%}. ")
        if difference['bootstrap_95'][0] <= 0:
            finding += 'This pilot does not demonstrate a transferred defense benefit.'
        lines[4:4] = [finding, '']
    if (OUT / 'runtime_amendment.json').exists():
        lines += ['## Documented runtime repair', '',
                  'The first GPU attempt passed the clean gate, then stopped on a tokenizer-boundary assertion. '
                  'The original evaluator, seal and interrupted outputs are preserved. A separately hashed wrapper '
                  'tokenizes the unchanged full prompt once and splits token offsets into memory chunks; only whitespace '
                  'tokens may cross a span boundary. '
                  'This implementation repair occurred after initial outcomes were available. Questions, notes, labels, '
                  'router, backbone, scoring, gates and success criteria were not changed. '
                  'See `runtime_amendment.json` and `interrupted_run/`.', '']
    recovery_path = OUT / 'execution_recovery.json'
    if result and recovery_path.exists():
        recovery = json.loads(recovery_path.read_text())
        if recovery.get('status') == 'complete':
            lines += ['The reported measurements come from the completed 2026-09-29 rerun. '
                      'An earlier repaired attempt completed, but its export exceeded the connector message limit '
                      'and the runtime became unavailable before the full artifacts were recovered. '
                      'The same sealed experiment was repeated without changing the inputs or evaluator; '
                      'small checkpoints were saved locally throughout. See `execution_recovery.json`, '
                      '`recovered_run_20260929/` and `result_audit.json`. This is a repeat of the same pilot, '
                      'not an independent dataset.', '']
    routing = result.get('router') if result else None
    if routing is None and (OUT / 'routing.json').exists():
        routing = json.loads((OUT / 'routing.json').read_text())
    if routing:
        rows = routing['records']
        attacks = [r for r in rows if r['malicious']]
        safe = [r for r in rows if not r['malicious']]
        lines += [
            '## CPU router diagnostic', '',
            'These are note-classification measurements against full abstracts, not Mamba answer accuracy. '
            'Per-condition routing can differ when the supporting sentence is removed.', '',
            f"- Malicious notes rejected: {sum(r['rejected'] for r in attacks)}/{len(attacks)}.",
            f"- Safe notes rejected: {sum(r['rejected'] for r in safe)}/{len(safe)}.",
            f"- Unknown question attribute: {routing['unknown_attribute']}/{len(rows)}.",
            f"- No trusted value extracted: {routing['no_trusted_value_extracted']}/{len(rows)}.", '',
        ]
    if result:
        gate = result['clean_gate']
        gain = gate['context_gain']
        lines += ['## Baseline gates', '',
                  f"Clean candidate accuracy: {rate(gate['clean'])}. Empty-context accuracy: {rate(gate['empty_context'])}.", '',
                  f"Paired context gain: {gain['difference']:+.1%}; paper-bootstrap 95% interval "
                  f"{gain['bootstrap_95'][0]:+.1%} to {gain['bootstrap_95'][1]:+.1%}. Clean gate passed: **{gate['passed']}**.", '']
        if 'attack_gate' in result:
            attack = result['attack_gate']
            lines += [f"Original late-attack accuracy: {rate(attack['attack_late'])}. Attack gate passed: **{attack['passed']}**.", '']
        if result['status'] != 'complete':
            lines += ['The predeclared gate stopped dependent defense-accuracy evaluation. No robustness gain is established.', '']
        if 'summary' in result:
            lines += ['## Answer accuracy', '',
                      '| Condition | Original ranked | Dual ranked | Original free generation | Dual free generation |',
                      '| --- | ---: | ---: | ---: | ---: |']
            for condition in ['clean', 'safe_addition', 'attack_early', 'attack_late', 'mixed_note']:
                a = result['summary']['original'][condition]
                b = result['summary']['dual_state'][condition]
                lines.append('| ' + condition + ' | ' + ' | '.join(rate(m) for m in
                             [a['candidate'], b['candidate'], a['free_generation'], b['free_generation']]) + ' |')
            lines += ['', 'Ranked accuracy sums log likelihood over each of two full completions. Free generation '
                      'uses a strict normalized answer-at-start parser; those metrics are separate.', '',
                      '## Controlled speed', '',
                      '| Variant | Full tokens/sec | Peak allocated bytes | Two-bank tensor capacity bytes |',
                      '| --- | ---: | ---: | ---: |']
            for name, summary in result['summary'].items():
                t = summary['timing']
                lines.append(f"| {name} | {t['full_tokens_per_second']:.2f} | {t['peak_allocated_bytes']} | {t['two_bank_capacity_bytes']} |")
            lines += ['', f"Dual/original full-path throughput ratio: {result['speed_ratio']:.3f}. "
                      'Eight fixed paper/condition pairs, two repeats, alternating variant order, '
                      '16 generated tokens and CUDA synchronization. Tensor capacity is not total peak memory.', '',
                      '## Predeclared criteria', '']
            lines += [f"- {name}: **{passed}**" for name, passed in result['criteria'].items()]
            lines.append('')
        lines += ['## Memory/filter equivalence', '']
        for name, checks in [('Forced-mask structural probes', result.get('mechanism_probes', [])),
                             ('Actual routing decisions', result.get('filtering_checks', []))]:
            if checks:
                lines += [f"{name}: {sum(r['prediction_matches'] for r in checks)}/{len(checks)} predictions and "
                          f"{sum(r['generated_tokens_match'] for r in checks)}/{len(checks)} generated-token sequences match. "
                          f"Maximum absolute candidate log-likelihood difference: {max(r['max_log_likelihood_error'] for r in checks):.6g}.", '']
        lines += [f"Backbone unchanged: **{result['backbone_after_sha256'] == result['manifest']['backbone_before_sha256']}**. "
                  f"GPU: {result['manifest']['gpu']}.", '']
    else:
        if interrupted_gate:
            gain = interrupted_gate['context_gain']
            lines += ['## Preserved interrupted-run measurements', '',
                      f"Clean candidate accuracy: {rate(interrupted_gate['clean'])}. "
                      f"Empty-context accuracy: {rate(interrupted_gate['empty_context'])}.", '',
                      f"Paired context gain: {gain['difference']:+.1%}; paper-bootstrap 95% interval "
                      f"{gain['bootstrap_95'][0]:+.1%} to {gain['bootstrap_95'][1]:+.1%}. "
                      f"Clean gate passed: **{interrupted_gate['passed']}**.", '',
                      'These measurements come from the original interrupted attempt in `interrupted_run/`. '
                      'They do not establish completed attack, defense or timing results, or verify the repaired run.', '']
        lines += ['## GPU evaluation pending', '',
                  'No completed repaired-run results are available locally. '
                  'The complete experiment requires a positive paired clean-versus-empty-context interval and '
                  'nonconstant candidate positions, then a late-attack accuracy drop. Failure stops the '
                  'dependent defense comparison without changing the tasks or router.', '']
    lines += ['## Reproduce', '',
              'Open `../generalization_mamba.ipynb` in a Colab T4 runtime and run top to bottom. '
              'It embeds the sealed data, code, router and historical hashes. '
              'No paid generation or automatic dependency installation occurs. '
              'Each run creates a new directory; stopped runs also produce a report and archive.', '',
              'This is a 32-paper pilot with synthetic notes, two candidates, and known note boundaries. '
              'It does not establish broad injection robustness or a capability beyond filtering rejected text. '
              'Do not retune this frozen router against these outcomes.', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines))
    print('Wrote generalization_experiment/REPORT.md')


if __name__ == '__main__':
    render()
