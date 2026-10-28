"""Post-hoc shortcut and score-margin audit. No new model inference or tuning."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import statistics

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def contains(text, value):
    return bool(re.search(r'(?<!\w)' + re.escape(value) + r'(?!\w)', text, re.I))


def margin(row):
    g = row['gold']
    return row['original_scores'][g] - row['original_scores'][1-g]


def main():
    protocol = read('final_hispa_audit/protocol.json')
    for name, expected in protocol['files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    result = read('final_hispa_audit/results.json')
    records = read('generalization_experiment/dataset.json')['records']
    selected = random.Random(protocol['selection_seed']).sample(records, 16)
    assert [r['paper_id'] for r in selected] == protocol['gpu_paper_ids']
    rows = result['rows']
    assert len(rows) == 288
    lookup = {(r['paper_id'], r['condition']): r for r in rows}
    assert len(lookup) == len(rows)
    by_id = {r['paper_id']: r for r in records}
    for row in rows:
        assert row['gold'] == by_id[row['paper_id']]['gold_index']
        for variant in ['original', 'dual']:
            scores = row[variant + '_scores']
            assert row[variant + '_correct'] == (max(range(2), key=scores.__getitem__) == row['gold'])
    tasks = []
    for r in selected:
        present = [contains(r['abstract'], c) for c in r['choices']]
        lexical_prediction = present.index(True) if sum(present) == 1 else None
        tasks.append(dict(paper_id=r['paper_id'], gold=r['gold_index'],
                          lexical_prediction=lexical_prediction,
                          lexical_correct=lexical_prediction == r['gold_index'],
                          answer_in_question=contains(r['stem'], r['answer']),
                          foil_in_question=contains(r['stem'], r['foil']),
                          answer_without_support=contains(r['context_without_support'], r['answer']),
                          empty_correct=lookup[r['paper_id'], 'empty_context']['original_correct'],
                          clean_margin=margin(lookup[r['paper_id'], 'clean']),
                          smallest_trigger_margin=min(margin(lookup[r['paper_id'], t]) for t in protocol['triggers'])))
    effects = {}
    for trigger in protocol['triggers']:
        pairs = [(lookup[r['paper_id'], trigger], lookup[r['paper_id'], 'benign_' + trigger]) for r in selected]
        assert all(a['span_token_count'] == b['span_token_count'] for a, b in pairs)
        changes = [margin(a) - margin(b) for a, b in pairs]
        effects[trigger] = dict(n=16, original_correct=sum(a['original_correct'] for a, _ in pairs),
                               dual_correct=sum(a['dual_correct'] for a, _ in pairs),
                               mean_attack_minus_benign_margin=statistics.mean(changes),
                               minimum_attack_minus_benign_margin=min(changes),
                               maximum_attack_minus_benign_margin=max(changes),
                               margins_lower_than_benign=sum(x < 0 for x in changes),
                               minimum_attack_margin=min(margin(a) for a, _ in pairs))
    # A retrospective matched-source comparison, not a new attack trial.
    historical = read('generalization_experiment/results.json')
    selected_ids = set(protocol['gpu_paper_ids'])
    comparison = {}
    for variant in ['original', 'dual']:
        comparison[variant] = {}
        for condition in ['clean', 'attack_early', 'attack_late', 'mixed_note']:
            group = [r for r in historical[variant] if r['paper_id'] in selected_ids and r['condition'] == condition]
            assert len(group) == 16
            comparison[variant][condition] = dict(n=16, correct=sum(r['correct'] for r in group))
    all_shortcut = sum(contains(r['abstract'], r['answer']) and not contains(r['abstract'], r['foil']) for r in records)
    tree = ast.parse((ROOT/'cloze_dual_state_eval.py').read_text())
    scorer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'candidate_scores')
    signature = [a.arg for a in scorer.args.args]
    assert signature == ['model', 'tokenizer', 'state', 'inputs', 'cache', 'candidates']
    audit = dict(status='passed', scope='Post-hoc saved-score and source audit; no new GPU inference; no causal attribution of the ceiling',
                 source_hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in [
                     'final_hispa_audit/results.json', 'final_hispa_audit/protocol.json',
                     'generalization_experiment/results.json', 'build_generalization_data.py',
                     'cloze_dual_state_eval.py', 'final_hispa_audit/evaluate.py']},
                 predictions_recomputed=576, selection_reproduced=True,
                 lexical_shortcut_correct=sum(t['lexical_correct'] for t in tasks),
                 lexical_shortcut_n=16, entire_parent_dataset_shortcut_correct=all_shortcut,
                 entire_parent_dataset_n=len(records), gold_position_counts=dict(Counter(r['gold_index'] for r in selected)),
                 question_contains_answer_or_foil=sum(t['answer_in_question'] or t['foil_in_question'] for t in tasks),
                 support_removal_leaves_literal_answer=sum(t['answer_without_support'] for t in tasks),
                 empty_context_correct=sum(t['empty_correct'] for t in tasks),
                 clean_margin_min=min(t['clean_margin'] for t in tasks),
                 trigger_margin_min=min(t['smallest_trigger_margin'] for t in tasks),
                 scorer_arguments=signature,
                 label_leakage_assessment='Code review found no gold index/attack label passed to the candidate scorer or router. Teacher-forced candidate likelihood uses each candidate as the prediction target with causal shift. This limited audit does not exclude pretraining overlap or all implementation bugs.',
                 semantic_review_limitation='The model-based dataset reviewer always saw the correct candidate first; its note IDs also used fixed ordering. These are annotation-review cues, not inputs to the evaluated Mamba scorer.',
                 effects=effects, matched_source_historical_comparison=comparison, tasks=tasks)
    (OUT/'hispa_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    lines = ['# Follow-up on the HiSPA ceiling', '',
             'October 16, 2026. Post-hoc analysis of retained results and source code; no new model inference.', '',
             '## Findings', '',
             '- The undefended model was evaluated: clean 16/16, every published trigger 16/16, and every matched benign condition 16/16. The controller accepted every trigger, so these are not successful rejection-based defenses.',
             f'- A literal-answer-presence rule scores {audit["lexical_shortcut_correct"]}/16 on the pilot and {all_shortcut}/32 on its parent dataset. The generator required the answer to occur in the source and the foil to be absent. This creates a shortcut; it does not establish that Mamba used it.',
             '- Removing the entire source still leaves 12/16 correct. Those prompts retain the cloze question and candidate completions; this is not a completely input-free baseline.',
             f'- Correct option positions are balanced: {dict(Counter(r["gold_index"] for r in selected))}. Neither candidate occurs verbatim in any of the 16 questions. Removing the support sentence removes the literal answer in all 16 cases.',
             '- The scoring call receives both candidate continuations but no correct-option index. Each candidate is scored as a separate continuation with a causal one-token shift; this is standard likelihood ranking, not unconstrained generation. The label is compared only after scoring. No direct gold-label leak was found on this reviewed path.',
             '- The semantic reviewer always saw the correct candidate first, with fixed note ordering. That weakens the claimed independence of annotation review and should be corrected in future datasets. It is distinct from a leak into the evaluated model.',
             f'- The smallest clean correct-versus-foil log-likelihood margin is {audit["clean_margin_min"]:.3f} nats; the smallest across all trigger cases is {audit["trigger_margin_min"]:.3f}. Triggers can change likelihoods without crossing the binary decision boundary. Ceiling accuracy alone hides that movement.', '',
             '## Trigger effects on the undefended model', '',
             'All rows retain 16/16 accuracy. A negative margin change means the trigger reduced the correct-answer margin relative to its matched-length benign control. These descriptive post-hoc margins are not calibrated probabilities or a new significance test.', '',
             '| Trigger | Mean attack − benign margin (nats) | Lower-margin cases | Minimum attacked margin |',
             '| --- | ---: | ---: | ---: |']
    for name, e in effects.items():
        lines.append(f'| {name} | {e["mean_attack_minus_benign_margin"]:+.3f} | {e["margins_lower_than_benign"]}/16 | {e["minimum_attack_margin"]:.3f} |')
    lines += ['', '## Existing attacks on the same source tasks', '',
              '| Historical condition | Undefended | Controller |', '| --- | ---: | ---: |']
    for condition in comparison['original']:
        lines.append(f'| {condition} | {comparison["original"][condition]["correct"]}/16 | {comparison["dual"][condition]["correct"]}/16 |')
    lines += ['', 'These historical synthetic notes and token serialization differ from the published-trigger pilot. This retrospective subset shows the task/model can be disrupted by some attacks; it is not a matched comparison of attack families or a reproduction of HiSPA.', '',
              '## What would resolve the remaining questions', '',
              'A fresh inference stress test should compare source removal, support-sentence removal, deterministic unrelated-source substitution, and counterfactual answer swaps; use the exact same 16 cases without selecting a favorable subset. Report complete candidate log-likelihoods, token counts, a clearly secondary length-normalized sensitivity check, and free generation separately. Repeat with published distractor placement and checkpoint settings before making a HiSPA replication claim. Counterfactual swaps must be reviewed for coherence without exposing gold labels to reviewers. These new inference controls have not been run in this follow-up.', '',
              'A new domain-transfer study must also use untouched final domains and split source/near-duplicate groups before constructing attacks. Previously inspected external tasks cannot become unseen tests merely by resetting a seed.', '',
              'Conclusion: retain the result as a failed attack-transfer pilot, with the shortcut disclosed. It supplies no evidence of successful defense. A source/code audit cannot rule out every possible data leak or pretraining overlap.']
    (OUT/'HISPA_REVIEW.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:audit[k] for k in ['status','lexical_shortcut_correct','entire_parent_dataset_shortcut_correct','question_contains_answer_or_foil','empty_context_correct','trigger_margin_min','matched_source_historical_comparison']}, indent=2))


if __name__ == '__main__':
    main()
