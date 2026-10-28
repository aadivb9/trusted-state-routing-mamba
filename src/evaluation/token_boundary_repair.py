"""Documented runtime repair; preserves the original sealed evaluator and data."""
from bisect import bisect_left
import hashlib
import json

import evaluate_generalization as evaluation


def token_ranges(parts, offsets):
    """Assign whole-prompt tokens to spans, carrying boundary whitespace forward.

    A token spanning a boundary must consist only of whitespace. This preserves
    the complete prompt's canonical tokenization without moving factual content
    between trusted text and incoming notes. Filter probes use these same chunks.
    """
    text = ''.join(parts)
    boundaries = []
    position = 0
    for part in parts:
        position += len(part)
        boundaries.append(position)
    buckets = [[] for _ in parts]
    for index, (start, end) in enumerate(offsets):
        for boundary in boundaries[:-1]:
            if start < boundary < end and not text[start:end].isspace():
                raise ValueError('A non-whitespace token crosses a trust boundary.')
        bucket = bisect_left(boundaries, end)
        buckets[bucket].append(index)
    ranges = []
    cursor = 0
    for indices in buckets:
        if indices:
            assert indices == list(range(cursor, cursor + len(indices)))
        ranges.append((cursor, cursor + len(indices)))
        cursor += len(indices)
    assert cursor == len(offsets)
    return ranges


def prepare(model, tokenizer, state, router, record, condition, defended=False,
            filter_mask=None, forced_mask=None):
    import torch
    context, names = evaluation.layout(record, condition)
    notes = ['\n\nExternal note: ' + record[name] for name in names]
    question = '\n\nAccording to the study, ' + record['stem'][0].lower() + record['stem'][1:]
    choices = [' ' + c + '.' for c in record['choices']]
    parts = [context, *notes, question]
    encoded = tokenizer(''.join(parts), add_special_tokens=False,
                        return_offsets_mapping=True, return_tensors='pt')
    ids = encoded['input_ids'].to(model.device)
    ranges = token_ranges(parts, encoded['offset_mapping'][0].tolist())
    chunks = [ids[:, start:end] for start, end in ranges]
    assert torch.equal(torch.cat(chunks, dim=1), ids)
    if not defended:
        selected = [chunks[0], *[c for i, c in enumerate(chunks[1:-1])
                                if filter_mask is None or filter_mask[i]], chunks[-1]]
        return torch.cat(selected, dim=1), None, choices, []
    route = evaluation.decisions(router, record, condition)
    if forced_mask is not None:
        assert len(forced_mask) == len(route)
        route = [{**r, 'rejected': not keep, 'forced_probe': True}
                 for r, keep in zip(route, forced_mask)]
    committed = state.capture(chunks[0])['cache']
    for note, decision in zip(chunks[1:-1], route):
        candidate = state.clone(committed)
        state.advance(note, candidate)
        if not decision['rejected']:
            committed = candidate
    question_ids = chunks[-1]
    if question_ids.shape[1] > 1:
        state.advance(question_ids[:, :-1], committed)
    return question_ids[:, -1:], committed, choices, route


def run():
    original_verify = evaluation.verify

    def verify():
        protocol = original_verify()
        amendment = json.loads((evaluation.OUT / 'runtime_amendment.json').read_text())
        for name, expected in amendment['files'].items():
            assert hashlib.sha256((evaluation.ROOT / name).read_bytes()).hexdigest() == expected, name
        return {**protocol, 'runtime_amendment': amendment}

    evaluation.verify = verify
    evaluation.prepare = prepare
    evaluation.run()


if __name__ == '__main__':
    run()
