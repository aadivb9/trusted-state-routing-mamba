"""Assign full-prompt tokens to observable regions without committing note bytes."""
from bisect import bisect_left

def serialize(prefix,spans,suffix,tokenizer):
 pieces=[prefix,*[s['text'] for s in spans],suffix];prompt=''.join(pieces);boundaries=[];pos=0
 for piece in pieces:pos+=len(piece);boundaries.append(pos)
 encoded=tokenizer(prompt,add_special_tokens=False,return_offsets_mapping=True);parts=[[] for _ in pieces]
 for token,(left,right) in zip(encoded['input_ids'],encoded['offset_mapping']):
  owners={bisect_left(boundaries,i+1) for i in range(left,right) if not prompt[i].isspace()}
  if len(owners)>1:raise ValueError('Non-whitespace token crosses observable region boundary')
  overlaps=[i for i in range(1,len(pieces)-1) if left<boundaries[i] and right>boundaries[i-1]]
  if overlaps and owners and next(iter(owners)) in [0,len(pieces)-1]:raise ValueError('Untrusted whitespace shares a trusted-content token')
  owner=next(iter(owners)) if owners else (overlaps[-1] if overlaps else bisect_left(boundaries,right))
  parts[owner].append(token)
 assert [v for p in parts for v in p]==encoded['input_ids']
 return prompt,parts

def retained(parts,rejected):
 assert len(parts)==len(rejected)+2
 return parts[0]+[v for chunk,drop in zip(parts[1:-1],rejected) if not drop for v in chunk]+parts[-1]
