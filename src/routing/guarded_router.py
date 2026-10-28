"""Version 1: conservative schema-bound evidence policy, with explicit abstention.

This is an unfitted engineering revision, not a demonstrated general detector.
Only trusted reference text, an incoming note, and a question enter inference.
"""
import re
from src.routing import context_router as historical

VERSION = 'guarded-symbolic-v1'
SMALL = dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(), range(20)))
TENS = dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(), range(20, 100, 10)))
WORDS = {**SMALL, **TENS, 'hundred': 100, 'thousand': 1000}
NUM = '|'.join(sorted(WORDS, key=len, reverse=True))
NUMBER_WORDS = re.compile(r'\b(?:' + NUM + r')(?:[\s-]+(?:and[\s-]+)?(?:' + NUM + r'))*\b', re.I)
DIRECTIVE = re.compile(r'\b(?:ignore|override|disregard|replace)\b.{0,90}\b(?:source|context|abstract|instructions?)\b|\b(?:answer|output|respond|return)\s+(?:only|with|instead|as)\b', re.I)
QUOTE = re.compile(r'"[^"\n]*"|“[^”\n]*”')
REJECTION = re.compile(r'\b(?:do not follow|do not adopt|example to reject|example to avoid|not an instruction|reject this|rejected example)\b', re.I)


def normalize_numbers(text):
    def convert(match):
        total = current = 0
        for word in re.split(r'[\s-]+', match.group().lower()):
            if word == 'and':
                continue
            if word == 'hundred':
                current = max(current, 1) * 100
            elif word == 'thousand':
                total += max(current, 1) * 1000
                current = 0
            else:
                current += WORDS[word]
        return str(total + current)
    return NUMBER_WORDS.sub(convert, text)


def attribute(question):
    q = question.casefold()
    if re.search(r'\b(?:hardware|processor|cpu|gpu)\b|\bran on\b', q):
        return 'hardware'
    if re.search(r'\b(?:method|fastest)\b', q):
        return 'method'
    if re.search(r'\b(?:repeat\w*|repetition\w*|runs?|trials?)\b', q):
        return 'runs'
    if re.search(r'\b(?:documents?|texts?|papers?|records?|corpus|evaluation set|sample size)\b', q):
        return 'documents'
    return None


def decision(action, reason, slot, source=(), claims=()):
    return {'version': VERSION, 'action': action, 'reason': reason,
            'commit_update': action == 'accept', 'risk': None, 'calibrated': False,
            'supported': slot is not None and bool(source),
            'evidence': {'attribute': slot, 'trusted_values': sorted(source), 'note_values': sorted(claims)}}


def route(trusted, note, question):
    slot = attribute(question)
    if not note.strip():
        result = decision('accept', 'empty_update', slot)
        result['commit_update'] = False
        return result
    if slot is None:
        return decision('abstain', 'unsupported_question', None)
    source = historical.values(normalize_numbers(trusted), slot, detailed=True,
                               study_ids=set(historical.STUDY.findall(trusted)))
    if len(source) != 1:
        return decision('abstain', 'missing_or_ambiguous_trusted_reference', slot, source)
    body = note
    rejected_quote = False
    if QUOTE.search(body):
        outside = QUOTE.sub('', body)
        if REJECTION.search(outside):
            body = outside
            rejected_quote = True
        else:
            return decision('abstain', 'unscoped_quotation', slot, source)
    normalized = normalize_numbers(body)
    for match in DIRECTIVE.finditer(normalized):
        before = normalized[max(0, match.start() - 16):match.start()]
        if not re.search(r'(?:do not|never)\s*$', before, re.I):
            return decision('reject', 'answer_or_source_override', slot, source)
    claims = historical.values(normalized, slot, detailed=True,
                               study_ids=set(historical.STUDY.findall(trusted)))
    if claims - source:
        return decision('reject', 'extracted_conflict', slot, source, claims)
    if claims:
        return decision('accept', 'extracted_agreement', slot, source, claims)
    if rejected_quote and re.fullmatch(r'[\s:.,;!]*(?:do not follow (?:this |the )?instruction|example to reject|example to avoid|not an instruction|reject this|rejected example)[\s:.,;!]*', body, re.I):
        return decision('accept', 'explicitly_rejected_quote_only', slot, source)
    return decision('abstain', 'no_verifiable_claim', slot, source)


def select_state(result, trusted_state, candidate_state):
    """Controller adapter: abstention always preserves committed state."""
    if result.get('action') not in ('accept', 'reject', 'abstain'):
        raise ValueError('Invalid routing action')
    if result.get('action') == 'accept' and result.get('commit_update') is True:
        return candidate_state
    return trusted_state
