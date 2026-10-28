"""Recompute October completion claims from raw responses and paired timings."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(root, name):
    return json.loads((root / name).read_text())


def require(value, reason):
    if not value:
        raise ValueError(reason)


def external(root):
    folder = root / 'experiments/17_external_validation'
    protocol = read(folder, 'protocol.json')
    repair = read(folder, 'runtime_amendment_v2.json')
    require(repair['previous_amendment_sha256'] == sha(folder / 'runtime_amendment.json'),
            'Runtime amendment chain changed')
    for name, digest in repair['files'].items():
        require(sha(root / name) == digest, 'Runtime repair input changed: ' + name)
    failed = read(folder, 'results.json')
    require(sha(folder / 'results.json') == repair['failed_attempt_sha256'], 'Failed attempt changed')
    require(failed['status'] == 'failed' and not failed['development'] and not failed['final'],
            'First attempt was not the retained pre-scoring runtime failure')
    result = read(folder, 'results_v2.json')
    require(result['protocol_sha256'] == sha(folder / 'protocol.json') and
            result['runtime_repair_sha256'] == sha(folder / 'runtime_amendment_v2.json') and
            result['runtime_amendment_sha256'] == sha(folder / 'runtime_amendment.json') and
            result['failed_attempt_sha256'] == sha(folder / 'results.json'), 'External result provenance mismatch')
    require(result['model'] == protocol['model'], 'External model changed')
    require(result['backbone_before'] == result['backbone_after'] and 'T4' in result['gpu'],
            'External weights or device check failed')
    base = root / 'archive/sealed_workspace'
    sys.path[:0] = [str(base), str(root)]
    from external_benchmark.audit_development import predictions
    prepared = read(folder, 'prepared/development.json')
    expected = {(r['dataset'], r['source_id']): r for r in prepared}
    seen = set()
    for row in result['development']:
        key = row['dataset'], row['source_id']
        require(key in expected and key not in seen, 'Unexpected or duplicate development source')
        seen.add(key)
        source = expected[key]
        prompt = source['instruction'] + '\nText: ' + source['text'] + '\n'
        require(row['source_prompt_sha256'] == hashlib.sha256(prompt.encode()).hexdigest(),
                'External prompt changed')
        strict, official = predictions(row['dataset'], row['response'])
        require(row['gold'] == source['label'] and row['strict'] == strict and row['official'] == official,
                'External saved score differs from raw response/source')
    require(seen == set(expected), 'Development coverage incomplete')
    gate = {}
    criteria = protocol['gate']
    for task in ('sst2', 'sms_spam', 'mrpc'):
        rows = [r for r in result['development'] if r['dataset'] == task]
        require(len(rows) == criteria['per_task_n'], 'Wrong competence-screen denominator')
        correct = sum(r['strict'] == r['gold'] for r in rows)
        valid = sum(r['strict'] is not None for r in rows)
        empty = predictions(task, result['instruction_only'][task]['response'])[0]
        gain = (correct - sum(empty == r['gold'] for r in rows)) / len(rows)
        gate[task] = {'strict_correct': correct, 'strict_valid': valid, 'input_gain': gain,
                      'passed': correct >= criteria['minimum_strict_correct'] and
                      valid >= criteria['minimum_strict_valid'] and
                      gain >= criteria['minimum_instruction_only_gain']}
    require(gate == result['gate'], 'External competence gate differs from raw rows')
    passed = all(g['passed'] for g in gate.values())
    require(not passed and result['gate_passed'] is False and result['status'] == 'stopped_clean_gate'
            and not result['final'], 'This closeout must retain the failed gate without efficacy inference')
    return {'gate': gate, 'development_generations': len(result['development']),
            'instruction_only_generations': len(result['instruction_only']), 'attack_generations': 0,
            'published_parser_correct': {task: sum(r['official']==r['gold'] for r in result['development']
                                                  if r['dataset']==task)
                                         for task in ('sst2','sms_spam','mrpc')}}


def annotation(root):
    folder = root / 'remediation/completion_20261027'
    protocol = read(folder, 'model_review_protocol.json')
    result = read(folder, 'model_review.json')
    for name, digest in protocol['files'].items():
        require(sha(root / name) == digest, 'Model-review code changed')
    packet_path = root / 'experiments/16_blinded_annotation/reviewer_packet.json'
    require(sha(packet_path) == protocol['packet_sha256'] and
            result['protocol_sha256'] == sha(folder / 'model_review_protocol.json') and
            result['code_sha256'] == sha(folder / 'review_annotations.py') and
            result['model_digest'] == protocol['model_digest'], 'Model-review provenance changed')
    require(result['status'] == 'complete' and result['reviewer']['human'] is False,
            'Model review must not become human evidence')
    packet = json.loads(packet_path.read_text())
    private = read(root, 'experiments/16_blinded_annotation/private/mapping.json')
    sys.path.insert(0, str(root))
    from src.validation.blinded_review import validate_reviews
    comparison = validate_reviews(packet, private, result['reviews'])
    require(comparison == result['comparison'], 'Annotation disagreements differ from raw labels')
    requests = [json.loads(line) for line in (folder / 'model_review_requests.jsonl').read_text().splitlines()]
    require(len(requests) == len(packet['items']) == len(result['reviews']) == 32,
            'Unexpected model-review request/review coverage')
    expected = {r['item_id']: r for r in result['reviews']}
    items = {r['item_id']: r for r in packet['items']}
    tree = ast.parse((folder / 'review_annotations.py').read_text())
    system = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
              and isinstance(n.value, str) and n.value.startswith('You are a source-grounded annotation reviewer.')]
    require(len(system) == 1, 'Unexpected reviewer prompt definition')
    require(len({r['item_id'] for r in requests}) == 32 and all(r['attempt'] == 0 for r in requests),
            'Unexpected repeated model-review attempts')
    for request in requests:
        require(json.loads(request['response']['message']['content']) == expected[request['item_id']],
                'Frozen model response differs from retained review')
        messages = [{'role': 'system', 'content': system[0]},
                    {'role': 'user', 'content': json.dumps(items[request['item_id']])}]
        require(request['request_sha256'] == hashlib.sha256(json.dumps(messages).encode()).hexdigest(),
                'Reviewer request differs from the blinded packet-only prompt')
        require(request['response']['model'] == protocol['model'] and request['response']['done'],
                'Unexpected/incomplete model-review response')
    require(all(r['candidate_reason'].strip() and all(n['reason'].strip() for n in r['notes'])
                for r in result['reviews']), 'Model review is missing source-grounded reasons')
    disagreements = comparison['disagreements']
    return {'items': len(result['reviews']), 'notes': sum(len(r['notes']) for r in result['reviews']),
            'note_disagreements': sum(r['type'] == 'note_label' for r in disagreements),
            'candidate_disagreements': sum(r['type'] == 'supported_candidate' for r in disagreements),
            'human_review_completed': False}


def timing(root):
    folder = root / 'experiments/18_runtime_validation'
    protocol = read(folder, 'timing_protocol.json')
    require(protocol['parent_protocol_sha256'] == sha(folder / 'protocol.json'), 'Timing parent changed')
    for name, digest in protocol['files'].items():
        require(sha(root / name) == digest, 'Timing input changed: ' + name)
    result = read(folder, 'timing_results.json')
    require(result['protocol_sha256'] == sha(folder / 'timing_protocol.json') and
            result['model'] == protocol['model'], 'Timing provenance mismatch')
    require(result['status'] == 'failed' and 'CUDA out of memory' in result['error']
            and 'backbone_after' not in result and not result.get('all_checks_passed', False),
            'The failed full timing protocol must not become a completed/passing run')
    require('T4' in result['gpu'] and result['parameter_dtypes'] == ['torch.float32'],
            'Timing device/precision changed')
    for name, version in protocol['runtime_versions'].items():
        require(result['versions'][name] == version, 'Timing runtime version changed')
    plan = protocol['plan']
    import math
    import random
    import statistics
    rng = random.Random(plan['paired_order_seed'])
    expected = {'warmups': [], 'measurements': []}
    for length in plan['prefix_tokens']:
        for phase, count in [('warmups', plan['warmups']), ('measurements', plan['timed_repetitions'])]:
            for rep in range(count):
                modes = list(plan['modes'])
                rng.shuffle(modes)
                expected[phase] += [(length, mode, rep) for mode in modes]
    for phase in expected:
        actual = [(r['prefix_tokens'], r['mode'], r['repetition']) for r in result[phase]]
        require(actual == expected[phase][:len(actual)], 'Timing rows differ from the frozen ordered plan')
        require(all(len(r['generated_tokens']) == plan['new_tokens'] for r in result[phase]),
                'Timing generated length changed')
    require(len(result['warmups']) == 36 and len(result['measurements']) == 166,
            'Recorded timing failure coverage changed')
    checks = result['cache_checks']
    require(all(checks[k] for k in ['separate_storage', 'trusted_unchanged', 'restore_matches',
                                   'generation_cache_handoff']) and
            checks['continuation_max_logit_error'] < 1e-3 and checks['chunk_vs_token_logit_error'] < 1e-3,
            'Pre-timing cache validation failed')
    # Describe only the completely observed shorter blocks. This does not pass
    # or truncate the original three-length protocol; that protocol remains failed.
    summaries, pairs = [], []
    for length in [128, 512]:
        groups = {}
        for mode in plan['modes']:
            rows = [r for r in result['measurements'] if r['prefix_tokens'] == length and r['mode'] == mode]
            require(sorted(r['repetition'] for r in rows) == list(range(20)), 'Short timing block incomplete')
            groups[mode] = {r['repetition']: r for r in rows}
            times = sorted(r['elapsed_seconds'] for r in rows)
            require(all(math.isfinite(t) and t > 0 for t in times), 'Invalid timing duration')
            summaries.append({'prefix_tokens': length, 'mode': mode, 'n': len(times),
                              'median_seconds': statistics.median(times),
                              'p95_seconds_nearest_rank': times[math.ceil(.95 * len(times)) - 1],
                              'peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in rows),
                              'peak_reserved_bytes': max(r['peak_reserved_bytes'] for r in rows)})
        for candidate, baseline in [('forced_accept', 'original'),
                                    ('forced_reject', 'identical_decision_filtering')]:
            ratios = [groups[candidate][rep]['elapsed_seconds'] / groups[baseline][rep]['elapsed_seconds']
                      for rep in range(20)]
            matched = sum(groups[candidate][rep]['generated_tokens'] == groups[baseline][rep]['generated_tokens']
                          for rep in range(20))
            require(matched == 20, 'Observed short-block token equivalence failed')
            pairs.append({'prefix_tokens': length, 'candidate': candidate, 'baseline': baseline,
                          'n_pairs': 20, 'matched_pairs': matched,
                          'median_paired_time_ratio': statistics.median(ratios)})
    return {'status': 'failed_cuda_out_of_memory', 'full_protocol_completed': False,
            'terminal_weight_hash_available': False, 'warmups': 36, 'timed': 166, 'planned_timed': 240,
            'completed_prefix_lengths': [128, 512], 'partial_prefix_length': 2048, 'partial_rows': 6,
            'equivalent_pairs': sum(p['matched_pairs'] for p in pairs),
            'summary': summaries, 'paired_comparisons': pairs,
            'scope': 'Post-failure descriptive summaries of completed shorter blocks only. The full protocol failed; no complete-run interval or terminal weight check is available.'}



def verify(root):
    return {'external': external(root), 'annotation': annotation(root), 'timing': timing(root)}


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parents[1]), indent=2))
