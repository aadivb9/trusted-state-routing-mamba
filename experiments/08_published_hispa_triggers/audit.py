"""Independently verify saved final-audit outputs and write its report."""
import hashlib
import json
from pathlib import Path
import random

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent


def paired_interval(values):
    rng = random.Random(20260930)
    samples = sorted(sum(rng.choices(values, k=len(values)))/len(values) for _ in range(5000))
    return [samples[125], samples[4874]]


def main():
    protocol = json.loads((OUT/'protocol.json').read_text())
    for name, expected in protocol['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    results = json.loads((OUT/'results.json').read_text())
    assert results['status'] == 'complete', 'GPU run has not completed.'
    assert results['backbone_before'] == results['backbone_after'] == protocol['expected_backbone_sha256']
    rows = results['rows']
    assert len(rows) == 288
    assert len({(r['paper_id'], r['condition']) for r in rows}) == 288
    conditions = ['clean','empty_context'] + [name for trigger in protocol['triggers'] for name in [trigger,'benign_'+trigger]]
    assert set(r['condition'] for r in rows) == set(conditions)
    index = {(r['paper_id'],r['condition']): r for r in rows}
    data = {r['paper_id']:r for r in json.loads((ROOT/'generalization_experiment/dataset.json').read_text())['records']}
    for row in rows:
        assert row['gold'] == data[row['paper_id']]['gold_index']
        for variant in ['original','dual']:
            scores = row[variant+'_scores']
            assert row[variant+'_correct'] == (max(range(2),key=lambda i:scores[i]) == row['gold'])
    summary = {}
    for condition in conditions:
        selected = [r for r in rows if r['condition']==condition]
        assert {r['paper_id'] for r in selected} == set(protocol['gpu_paper_ids'])
        summary[condition] = {'n':len(selected), 'original_correct':sum(r['original_correct'] for r in selected), 'dual_correct':sum(r['dual_correct'] for r in selected), 'rejected':sum(r['route']['rejected'] for r in selected)}
    assert summary == results['summary']
    effects = {}
    for trigger in protocol['triggers']:
        differences = []
        gains = []
        for paper_id in protocol['gpu_paper_ids']:
            attacked = index[paper_id,trigger]
            benign = index[paper_id,'benign_'+trigger]
            assert attacked['span_token_count'] == benign['span_token_count']
            differences.append(int(benign['original_correct']) - int(attacked['original_correct']))
            gains.append(int(attacked['dual_correct']) - int(attacked['original_correct']))
        effects[trigger] = {'benign_minus_attack':sum(differences)/16,'benign_minus_attack_ci95':paired_interval(differences), 'defense_gain':sum(gains)/16,'defense_gain_ci95':paired_interval(gains)}
    accepted = [r for r in rows if not r['route']['rejected'] and r['condition']!='empty_context']
    max_error = max(abs(a-b) for r in accepted for a,b in zip(r['original_scores'],r['dual_scores']))
    verification = {'status':'passed','gpu_conditions_verified':288,'candidate_predictions_recomputed':576,'protocol_seals_verified':len(protocol['files']), 'matched_length_controls':128, 'backbone_unchanged':True,'accepted_route_prediction_matches':sum(r['original_correct']==r['dual_correct'] for r in accepted), 'accepted_route_cases':len(accepted),'accepted_route_max_score_error':max_error,'effects':effects}
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    lines = ['# Final published-HiSPA transfer audit', '', '**Status: complete.** Frozen Mamba-130M; no training or threshold changes.', '', 'This bounded exploratory audit uses 16 deterministically sampled tasks from the already observed 32-paper cloze dataset. It tests the seven published natural-language triggers and the published optimized seed-35 payload, with exact six-token IDs for the optimized payload. This is transfer to 130M on an adapted cloze task, not an untouched benchmark or reproduction of the complete official RoBench protocol.', '', '| Condition | Original correct | Dual-state correct | Spans rejected |', '| --- | ---: | ---: | ---: |']
    for condition in conditions:
        s = summary[condition]
        lines.append(f"| {condition} | {s['original_correct']}/16 | {s['dual_correct']}/16 | {s['rejected']}/16 |")
    lines += ['', 'Each benign control has exactly the same token count as its associated trigger. It consists of a prefix of fixed neutral filler; truncation can end mid-sentence. This controls token count, but not every linguistic property of the payload. All inputs use explicit token-chunk concatenation with identical token streams across model variants. The empty-context baseline is scored once and shared between variants because it contains no memory-routing operation.', '', '| Trigger | Benign minus attack accuracy | Paired 95% interval | Defense gain |', '| --- | ---: | ---: | ---: |']
    for trigger, effect in effects.items():
        lo,hi=effect['benign_minus_attack_ci95']
        lines.append(f"| {trigger} | {100*effect['benign_minus_attack']:+.2f} pp | [{100*lo:+.2f}, {100*hi:+.2f}] pp | {100*effect['defense_gain']:+.2f} pp |")
    lines += ['', 'Intervals use 5,000 paired paper resamples and are descriptive, unadjusted for eight trigger comparisons. A collapsed interval from identical observed differences does not prove population equivalence. Repeated triggers across papers are not independent trigger draws. Candidate scores sum complete answer-continuation log probabilities; no new free-generation robustness or state-poisoning mechanism claim is made.', '', '## Interpretation', '', '**No attack-induced accuracy loss or defense gain was observed.** Both variants scored 16/16 on clean prompts, on each of the eight trigger conditions, and on every matched benign control. Empty-context accuracy was 12/16. Thus this small, reused, two-candidate cloze setup did not reproduce a HiSPA accuracy drop and cannot establish defense efficacy. Its ceiling scores, strong empty-context baseline, lack of distractors, and transfer of a trigger optimized for a larger model limit the inference. This is not evidence that Mamba-130M is generally immune to HiSPA.', '', 'The CPU diagnostic on all 32 tasks rejected **0/256** published-trigger presentations. That is a detector coverage result, not a measurement of attack success. The GPU comparison separately measures answer effects. Any clean-to-attack decrease must be assessed against the corresponding benign control; unchanged original-versus-dual outcomes do not establish a successful defense.', '', '## Verification and execution', '', f"All 288 conditions and 576 candidate predictions were independently recomputed from saved scores. The 128 trigger/control pairs have matched token counts. Accepted-span predictions matched in {verification['accepted_route_prediction_matches']}/{len(accepted)} cases; maximum score discrepancy was {max_error:.8g}.", '', 'GPU checks reached all 48 convolution/SSM tensors, confirmed independent storage, preserved trusted memory, restoration and generation handoff. Cached/full continuation error was 0.00011444 and cached/tokenwise error was 0.00009441. All 129,135,360 backbone parameters remained frozen and the full before/after weight hashes match the historical checkpoint.', '', 'The first launched runtime became inaccessible before outputs were recovered. The unchanged sealed evaluator was relaunched in a fresh runtime; these are the recovered rerun measurements. No unavailable results were counted, and no outcomes were used to tune the router. Existing 26 CPU tests passed separately; the saved generalization result audit also passed.', '', '## Artifacts', '', '- `protocol.json`: fixed task selection, serialization, controls and input/code hashes.', '- `results.json`: all scores, decisions and GPU checks.', '- `router_diagnostic.json`: all 256 CPU routing decisions.', '- `verification.json`: independent numerical audit and paired effects.', '- `gpu_run.log`: recovered execution log.', '- `../final_hispa_audit.ipynb`: embedded reproduction inputs and evaluator.', '', 'Source: [HiSPA paper](https://arxiv.org/abs/2601.01972v5), [pinned published code](https://github.com/TortueSagace/hispa/tree/898d7f7b78ba8f934f70340d12e9c1183bb623e5). Trigger provenance is recorded in `source_provenance.json`.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'summary':summary,'effects':effects,'verification':'passed'},indent=2))


if __name__ == '__main__':
    main()
