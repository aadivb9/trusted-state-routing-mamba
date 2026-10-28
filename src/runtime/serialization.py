"""Canonical token serialization shared by native filtering and cache selection."""
from bisect import bisect_left


def chunks(instruction,text,tokenizer):
    pieces=[instruction+'\nText: ',text,'\n']
    prompt=''.join(pieces)
    encoded=tokenizer(prompt,add_special_tokens=False,return_offsets_mapping=True)
    boundaries=[];position=0
    for part in pieces:position+=len(part);boundaries.append(position)
    output=[[] for _ in pieces]
    for token,(left,right) in zip(encoded['input_ids'],encoded['offset_mapping']):
        for boundary in boundaries[:-1]:
            if left<boundary<right and not prompt[left:right].isspace():
                raise ValueError('Non-whitespace token crosses trust boundary')
        output[bisect_left(boundaries,right)].append(token)
    assert [t for group in output for t in group]==encoded['input_ids']
    return prompt,output


def selected_tokens(parts,keep_note):
    prefix,note,suffix=parts
    return prefix+(note if keep_note else [])+suffix
