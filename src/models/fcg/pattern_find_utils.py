from .cx import Construction
from difflib import SequenceMatcher


def generalize_cxs(x0, y0, x1, y1) -> Construction:
    f_matches: list[tuple[int, int, int]] = list(SequenceMatcher(a=x0, b=x1).get_matching_blocks())
    m_matches: list[tuple[int, int, int]] = list(SequenceMatcher(a=y0, b=y1).get_matching_blocks())

    if len(f_matches) <= 1 or len(m_matches) <= 1:
        raise ValueError('no alignment found, refusing to generalize from the cx pairs.')

    form = matches_to_parts(f_matches, x0, x1)
    meaning = matches_to_parts(m_matches, y0, y1)
    cx = Construction(form, meaning, 'item')
    return cx


def matches_to_parts(matches: list[tuple[int, int, int]], a, b) -> list[str]:
    pieces: list[str] = []
    slot_id = 0

    for i, (oa, ob, l) in enumerate(matches):  # offset of a and b, and the block length
        if i == 0 and (oa > 0 or ob > 0):
            pieces.append(f'<{slot_id}>')
            slot_id += 1
        elif i > 0:
            last_oa, last_ob, last_l = matches[i - 1]
            if oa > last_oa + last_l or ob > last_ob + last_l:
                pieces.append(f'<{slot_id}>')
                slot_id += 1
        if l > 0:
            pieces.append(a[oa:oa + l].strip())

    return pieces
