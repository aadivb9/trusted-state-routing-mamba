"""Full-prompt tokenization with explicit ownership of boundary whitespace.

A token may contain padding from another region, but must never contain
non-whitespace characters from two different trust regions.
"""
from bisect import bisect_left


def chunks(instruction, text, tokenizer):
    pieces = [instruction + '\nText: ', text, '\n']
    prompt = ''.join(pieces)
    encoded = tokenizer(prompt, add_special_tokens=False,
                        return_offsets_mapping=True)
    boundaries = []
    position = 0
    for part in pieces:
        position += len(part)
        boundaries.append(position)
    output = [[] for _ in pieces]
    for token, (left, right) in zip(encoded['input_ids'],
                                   encoded['offset_mapping']):
        owners = {bisect_left(boundaries, index + 1)
                  for index in range(left, right)
                  if not prompt[index].isspace()}
        if len(owners) > 1:
            raise ValueError('Non-whitespace token crosses trust boundary')
        owner = next(iter(owners)) if owners else bisect_left(boundaries, right)
        output[owner].append(token)
    assert [token for group in output for token in group] == encoded['input_ids']
    return prompt, output


def selected_tokens(parts, keep_note):
    prefix, note, suffix = parts
    return prefix + (note if keep_note else []) + suffix
