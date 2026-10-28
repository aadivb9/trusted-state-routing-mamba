"""Trainable two-stage evidence router for the controlled Mamba fact task.

Only context, external text, and the question enter inference. Gold answers,
candidate lists, attack labels, record IDs, and family tags are never features.
This is a task-specific symbolic evidence extractor plus learned classifiers,
not a general entailment model or an internally trained Mamba architecture.
"""
import math
import re

FEATURE_NAMES = [
    'context_known', 'note_has_value', 'agreement', 'conflict',
    'only_conflict', 'agreement_and_conflict', 'missing_context_evidence',
    'negation', 'quotation', 'answer_directive', 'source_deference',
    'unknown_attribute', 'multiple_note_values', 'other_study',
]


def attribute(question):
    text = question.lower()
    if 'document' in text or 'evaluation set' in text or 'corpus' in text or 'evaluated' in text:
        return 'documents'
    if 'fastest' in text or 'method' in text:
        return 'method'
    if 'repeat' in text or 'repetition' in text or 'runs' in text:
        return 'runs'
    if 'hardware' in text or 'ran on' in text or 'comput' in text or 'processor' in text:
        return 'hardware'
    return None


PATTERNS = {
    'documents': [r'\b(\d+)\s+(?:documents?|texts?|papers?|records?)\b',
                  r'\b(?:document count|corpus size|evaluation set size)\s*(?:was|is|:|=|of)?\s*(\d+)\b'],
    'runs': [r'\b(\d+)\s+(?:repetitions?|times|runs|trials|replicates)\b',
             r'\b(?:run count|repetition count)\s*(?:was|is|:|=|of)?\s*(\d+)\b'],
    'hardware': [r'\b(CPU|GPU)\b'],
    'method': [r'\bMethod\s+([A-Z])\b'],
}
NEGATION = re.compile(r'\b(?:not|never|incorrect|false|erroneous|reject|disregard|ignore)\b', re.I)
DIRECTIVE = re.compile(r'\b(?:ignore|override|replace|disregard)\b.{0,55}\b(?:source|context|abstract|instructions?)\b|\b(?:answer|output|respond|return)\s+(?:only|with|instead|as)\b', re.I)
DEFERENCE = re.compile(r'\b(?:retain|preserve|follow|use|trust)\b.{0,35}\b(?:source|abstract|original|trusted)\b', re.I)
STUDY = re.compile(r'\b[A-Z][A-Za-z]+-[A-Z]?\d+\b')


def values(text, slot, detailed=False, study_ids=()):
    if slot is None:
        return set()
    found = set()
    sentences = re.split(r'(?<=[.!?;])["”]?\s+|\n', text)
    for sentence in sentences:
        ids = set(STUDY.findall(sentence))
        if detailed and ids and study_ids and not ids.intersection(study_ids):
            continue
        # Quoted instructions explicitly presented as examples to reject are
        # not factual assertions. A quotation alone is not treated as safe.
        quoted_example = (bool(re.search(r'["“”]', sentence)) and
                          bool(re.search(r'\b(?:reject|do not follow|do not adopt|example to avoid|not an instruction)\b', sentence, re.I)))
        for pattern in PATTERNS[slot]:
            for match in re.finditer(pattern, sentence, re.I):
                if detailed and quoted_example:
                    continue
                before = sentence[max(0, match.start()-45):match.start()]
                after = sentence[match.end():match.end()+32]
                # Local scope: "not X but Y" keeps Y, and "X is false" drops X.
                local_before = re.split(r'\b(?:but|instead|rather|however)\b|[,;:]', before, flags=re.I)[-1]
                negated = (bool(re.search(r'\b(?:not|never|without|incorrectly|erroneously)\b[^.!?;]*$', local_before, re.I)) or
                           bool(re.match(r'\s*(?:is|was|would be|are)\s+(?:false|incorrect|wrong|not correct)\b', after, re.I)))
                if not detailed or not negated:
                    found.add(match[1].upper())
    return found


def features(context, note, question, detailed=False):
    slot = attribute(question)
    study_ids = set(STUDY.findall(context))
    source = values(context, slot, detailed=True, study_ids=study_ids)
    claims = values(note, slot, detailed=detailed, study_ids=study_ids)
    agree = bool(source & claims)
    conflict = bool(source and claims-source)
    negation = bool(NEGATION.search(note))
    quotation = bool(re.search(r'["“”]', note))
    directive = bool(DIRECTIVE.search(note))
    deference = bool(DEFERENCE.search(note))
    other = bool(set(STUDY.findall(note))-study_ids)
    vector = [bool(source), bool(claims), agree, conflict,
              conflict and not agree, conflict and agree, bool(claims) and not source,
              negation, quotation, directive, deference, slot is None,
              len(claims)>1, other]
    return [float(v) for v in vector], {
        'attribute':slot, 'trusted_values':sorted(source), 'note_values':sorted(claims),
        'negation':negation, 'quotation':quotation, 'other_study':other,
    }


def sigmoid(x):
    return 1/(1+math.exp(-max(-40,min(40,x))))


def probability(head, vector):
    return sigmoid(head['bias']+sum(w*x for w,x in zip(head['weights'],vector)))


def route(router, context, note, question):
    if not note.strip():
        return {'rejected':False,'risk':0.,'stage':'empty','evidence':{}}
    vector, evidence = features(context,note,question)
    risk = probability(router['fast'],vector)
    low, high = router['uncertainty_band']
    refine = (low<risk<high or evidence['negation'] or evidence['quotation'] or
              evidence['other_study'] or len(evidence['note_values'])>1)
    if refine:
        vector, evidence = features(context,note,question,detailed=True)
        risk = probability(router['detailed'],vector)
    return {'rejected':risk>=router['threshold'],'risk':risk,
            'stage':'detailed' if refine else 'fast','evidence':evidence}
