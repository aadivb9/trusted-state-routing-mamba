"""Describe the measured adaptive-router result without changing its scorer."""
import json
from pathlib import Path
import adaptive_memory_router as routing
from dual_state_mamba import binary_metrics

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'adaptive_router_experiment'


def percent(x):return f'{100*x:.1f}%'


def build():
    r=json.loads((OUT/'results.json').read_text())
    router=json.loads((OUT/'router.json').read_text())
    lines=['# Adaptive context routing for Mamba-130m','',
        f"The independent 200-record test improved attacked candidate-ranking accuracy from {percent(r['summary']['original']['malicious']['candidate']['accuracy'])} "
        f"to {percent(r['summary']['dual_state']['malicious']['candidate']['accuracy'])}. "
        f"All predeclared criteria passed: **{r['all_criteria_passed']}**.",'',
        'Two frozen-Mamba variants were compared: original Mamba and Mamba with two cache banks selected by a retrained context-aware router. '
        'The Mamba backbone was not fine-tuned. The two logistic classifiers together have 30 trainable parameters.','',
        '## What changed','',
        'The previous router classified note wording without reading the source. This router compares values extracted from the trusted context and incoming note for the attribute asked about. '
        'It escalates uncertain cases, negations, quotations, multi-value notes, and references to other studies to a more detailed clause-level check. '
        'A learned classifier selects the trusted or candidate memory bank. Quotation alone is not treated as harmless. '
        'No gold answer, candidate list, attack label, record ID, or family tag is passed to routing.','',
        '**Scope:** this is adaptive evidence checking for a restricted synthetic fact schema, not general-purpose reasoning or a learned internal dual-SSM architecture. '
        'It uses lexical patterns for document counts, hardware, methods, and repetitions. Ambiguous language and unsupported attributes remain limitations.','',
        '## Training and separation','',
        '1,000 new controlled records trained the router; 200 separate records selected classifier epochs and the decision threshold. '
        'A further 200 controlled records were held out. Half the source groups use identical note wording under two different trusted contexts, with opposite labels. '
        'Pairs stay within a split. Controlled splits share a fact grammar and therefore are not strong evidence of language generalization.','',
        'After freezing the router, Claude generated 200 additional records across 100 new source groups. '
        'A separate label-blinded Claude audit checked all of them before evaluation. The same provider generated and reviewed the notes; there was no full human annotation. '
        'The previously observed failed benchmark is reported separately as a regression set, not an unseen test. '
        'Its failure informed this design, but its records were not used to fit parameters or select thresholds.','',
        '## Independent evaluation','',
        'Candidate ranking compares the summed conditional likelihood of two full answer completions. '
        'Free generation uses the unchanged strict value-at-start parser; valid facts appearing later in explanations still count as wrong.','',
        '| Metric | Original | Adaptive dual state |','| --- | ---: | ---: |']
    for condition in ['clean','safe','malicious']:
        for key in ['candidate','free_generation']:
            cells=[r['summary'][v][condition][key] for v in ['original','dual_state']]
            lines.append(f"| {condition}, {key} | "+' | '.join(f"{m['correct']}/{m['n']} ({percent(m['accuracy'])})" for m in cells)+' |')
    delta=r['paired_differences']['malicious']
    lines+=['',f"Attacked candidate gain: **{100*delta['difference']:+.1f} percentage points**, paired-bootstrap 95% interval "
            f"**{100*delta['bootstrap_95'][0]:+.1f} to {100*delta['bootstrap_95'][1]:+.1f} points** (5,000 source-group resamples).",'',
            '| Useful safe evidence (20 notes) | Original | Adaptive dual state |','| --- | ---: | ---: |']
    for key in ['candidate','free_generation']:
        cells=[r['summary'][v]['required_evidence'][key] for v in ['original','dual_state']]
        lines.append('| '+key+' | '+' | '.join(f"{m['correct']}/{m['n']}" for m in cells)+' |')
    lines+=['','## Routing','',
            '| Set | Accuracy | Attacks caught | Safe notes rejected | Detailed check |',
            '| --- | ---: | ---: | ---: | ---: |']
    sets={**{'controlled_'+k:v for k,v in r['router_controlled'].items()},
          'independent':r['router_independent'],'observed_regression':r['router_regression']}
    for name,values in sets.items():
        m=values['metrics']
        lines.append(f"| {name} | {percent(m['accuracy'])} | {m['tp']}/{m['tp']+m['fn']} | {m['fp']}/{m['fp']+m['tn']} | {percent(values['detailed_fraction'])} |")
    independent=json.loads((OUT/'independent_benchmark/dataset.json').read_text())['records']
    labels=[x['attack_label']=='malicious' for x in independent]
    fast=binary_metrics(labels,[routing.probability(router['fast'],routing.features(x['trusted_context'],x['span'],x['spec']['stem'])[0])>=router['threshold'] for x in independent])
    empty=binary_metrics(labels,[routing.route(router,'',x['span'],x['spec']['stem'])['rejected'] for x in independent])
    lines+=['',f"Post-freeze router diagnostics: using only the fast context classifier scores {percent(fast['accuracy'])}; "
            f"removing trusted context scores {percent(empty['accuracy'])}. These checks do not retrain anything or introduce another language-model variant. "
            'Fast-only performance matches the cascade here: the evidence supports context-aware verification, but does not demonstrate an additional accuracy benefit from adaptive escalation.','',
            'The frozen router missed a table-style document count it could not extract. That record was retained and the grammar was not patched after seeing it.']
    lines+=['','## Previously observed failure set','',
        'The old text-only defense scored 29% under attack on this set. The original rows below are reused from that saved run; '
        'adaptive rows were newly evaluated. Hashes verify the same Mamba weights, with unchanged prompt and scoring. This is a regression comparison.','',
        '| Candidate accuracy | Original | Adaptive dual state |','| --- | ---: | ---: |']
    for condition in ['clean','safe','malicious']:
        lines.append('| '+condition+' | '+' | '.join(percent(r['regression_summary'][v][condition]['candidate']['accuracy']) for v in ['original','dual_state'])+' |')
    lines+=['','## Generation speed','',
        'Twenty fixed independent-test records, three repetitions, alternating order, 16 generated tokens per call, warm-up and CUDA synchronization. '
        'Full timing includes routing, tokenization, extraction, copying, span processing, prefill and generation. Candidate scoring is separate. '
        'Both variants use last-token logits. Generation-only timing is not directly comparable because original-model prefill occurs inside `.generate()`, '
        'while dual state performs most prefill before that call.','',
        '| Full path | Original | Adaptive dual state |','| --- | ---: | ---: |',
        '| Tokens/sec | '+' | '.join(f"{r['summary'][v]['timing']['end_to_end_tokens_per_second']:.2f}" for v in ['original','dual_state'])+' |',
        '| `.generate()` tokens/sec (different prefill scope) | '+' | '.join(f"{r['summary'][v]['timing']['generation_only_tokens_per_second']:.2f}" for v in ['original','dual_state'])+' |',
        '',f"Dual/original throughput ratio: **{r['speed_ratio']:.3f}**. This is a batch-one implementation measurement on {r['manifest']['gpu']}, not an architectural speed guarantee.",'',
        '## Predeclared criteria','',
        '| Criterion | Passed |','| --- | --- |']
    lines.extend(f'| {key} | {value} |' for key,value in r['criteria'].items())
    lines+=['',f"All criteria passed: **{r['all_criteria_passed']}**.",'',
        '## Reproduction and limits','',
        '- Run `../adaptive_mamba.ipynb` on a Colab T4. No additional model or paid generation is required for reproduction.',
        '- Training used deterministic templates, not temperature sampling. Independent notes used Claude CLI, which does not expose temperature.',
        '- The original HiSPA extractor still supplies all convolution/SSM state. State isolation and restoration checks were executed on GPU.',
        '- Both banks share pretrained weights. This single-span controller can still be expressed as filtering rejected text; no additional capability over that filter is established.',
        '- Known trust boundaries and synthetic controlled facts are assumed. This does not establish optimized HiSPA resistance or broad natural-language robustness.',
        f"- Unchanged backbone SHA-256: `{r['backbone_after_sha256']}`.",
        f"- Router training took {router['training_seconds']:.3f} CPU seconds. No heavy dependencies or other model checkpoints were installed.",
        '- Raw notes, generation/audit calls, training data, router weights, protocol, evaluation seal, answers and timing measurements are saved in this directory.',
        '- New test outcomes were not used to retune this router. Future revisions must keep these outcomes as historical evaluations.',
    ]
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Wrote',OUT/'REPORT.md')


if __name__=='__main__':build()
