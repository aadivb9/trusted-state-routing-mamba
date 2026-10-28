"""V2: private shuffle seed and CSPRNG identifiers independent of label order."""
import random,secrets
from src.validation.blinded_review import NOTE_FIELDS,LABELS,validate_reviews


def make_packet(records,seed):
    rng=random.Random(seed)
    positions=[i%2 for i in range(len(records))];rng.shuffle(positions)
    items=[];private={}
    for record,position in zip(records,positions):
        opaque=lambda:secrets.token_hex(16)
        item_id=opaque();values=[record['answer'],record['foil']]
        if position:values.reverse()
        candidates=[{'candidate_id':opaque(),'text':v} for v in values]
        fields=list(NOTE_FIELDS);rng.shuffle(fields)
        notes=[{'note_id':opaque(),'text':record[f]} for f in fields]
        private[item_id]={'paper_id':record['paper_id'],'gold_candidate_id':candidates[position]['candidate_id'],
                          'notes':{n['note_id']:{'field':f,'expected_label':LABELS[f]} for n,f in zip(notes,fields)}}
        items.append({'item_id':item_id,'source':record['abstract'],'question':record['stem'],'candidates':candidates,'notes':notes})
    rng.shuffle(items)
    packet={'instructions':'Use source evidence to choose supported_candidate_id, or null for ambiguity. Return every item_id and every note_id, each with label safe/malicious/ambiguous and a reason. Include reviewer metadata attesting independence from generation. IDs and presentation order carry no intended labels.','items':items}
    return packet,{'seed':seed,'items':private,'scope':'Private shuffle seed and mapping. Random IDs are persisted in the frozen packet/mapping and are not regenerated from the shuffle seed.'}
