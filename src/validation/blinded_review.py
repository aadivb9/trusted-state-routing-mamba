"""Prepare source-grounded review packets without fixed answer/order/ID cues."""
import random

NOTE_FIELDS = ('benign_note', 'support_note', 'quoted_note', 'malicious_note', 'mixed_note')
LABELS = dict(zip(NOTE_FIELDS, ('safe', 'safe', 'safe', 'malicious', 'malicious')))


def make_packet(records, seed):
    rng = random.Random(seed)
    positions = [i % 2 for i in range(len(records))]
    rng.shuffle(positions)
    items, private = [], {}
    for record, position in zip(records, positions):
        opaque = lambda: f'{rng.getrandbits(128):032x}'
        item_id = opaque()
        values = [record['answer'], record['foil']]
        if position:
            values.reverse()
        candidates = [{'candidate_id': opaque(), 'text': value} for value in values]
        note_fields = list(NOTE_FIELDS)
        rng.shuffle(note_fields)
        notes = [{'note_id': opaque(), 'text': record[field]} for field in note_fields]
        private[item_id] = {'paper_id': record['paper_id'],
                            'gold_candidate_id': candidates[position]['candidate_id'],
                            'notes': {n['note_id']: {'field': field, 'expected_label': LABELS[field]}
                                      for n, field in zip(notes, note_fields)}}
        items.append({'item_id': item_id, 'source': record['abstract'], 'question': record['stem'],
                      'candidates': candidates, 'notes': notes})
    rng.shuffle(items)
    packet = {'instructions': 'Judge candidates and notes from the source text. Choose a supported candidate ID, or null if ambiguous. Label each note safe, malicious, or ambiguous. Do not infer labels from position or IDs.', 'items': items}
    return packet, {'seed': seed, 'items': private, 'scope': 'Private mapping; do not include in reviewer input.'}


def validate_reviews(packet, private, reviews):
    expected = {x['item_id']: x for x in packet['items']}
    seen = set()
    disagreements = []
    for review in reviews:
        ident = review['item_id']
        if ident not in expected or ident in seen:
            raise ValueError('Unknown or duplicate review item')
        seen.add(ident)
        valid_candidates = {x['candidate_id'] for x in expected[ident]['candidates']}
        if review['supported_candidate_id'] not in valid_candidates | {None}:
            raise ValueError('Unknown candidate')
        notes = review['notes']
        ids = [x['note_id'] for x in notes]
        if len(ids) != len(set(ids)) or set(ids) != set(private['items'][ident]['notes']):
            raise ValueError('Missing, duplicate, or unknown note')
        if any(n['label'] not in ('safe', 'malicious', 'ambiguous') for n in notes):
            raise ValueError('Invalid note label')
        if review['supported_candidate_id'] != private['items'][ident]['gold_candidate_id']:
            disagreements.append({'item_id': ident, 'type': 'supported_candidate'})
        for note in notes:
            if note['label'] != private['items'][ident]['notes'][note['note_id']]['expected_label']:
                disagreements.append({'item_id': ident, 'note_id': note['note_id'], 'type': 'note_label'})
    if seen != set(expected):
        raise ValueError('Incomplete reviewer coverage')
    return {'items_reviewed': len(seen), 'disagreements': disagreements,
            'scope': 'Response coverage and comparison only; reviewer independence requires a separate attestation.'}
