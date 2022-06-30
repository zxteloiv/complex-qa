from functools import partial
from typing import Optional, List
import re


def make_human_readable_text(output_tensor,
                             vocab,
                             ns: str,
                             stop_ids: Optional[list] = None,
                             get_token_from_id_fn=None,
                             ) -> list:
    """Convert the predicted word ids into discrete tokens"""
    import numpy as np
    import torch
    from trialbot.data import NSVocabulary
    output_tensor: torch.Tensor
    vocab: NSVocabulary

    # output_tensor: (batch, max_length)
    if not isinstance(output_tensor, np.ndarray):
        output_tensor = output_tensor.detach().cpu().numpy()

    stop_ids = [] if stop_ids is None else stop_ids

    get_token_from_id_fn = get_token_from_id_fn or partial(vocab.get_token_from_index, namespace=ns)

    all_readable_tokens = []
    for token_ids in output_tensor:
        if token_ids.ndim > 1:
            token_ids = token_ids[0]

        token_ids = list(token_ids)
        for stop_tok in stop_ids:
            if stop_tok in token_ids:
                token_ids = token_ids[:token_ids.index(stop_tok)]

        tokens = [get_token_from_id_fn(token_id) for token_id in token_ids]
        all_readable_tokens.append(tokens)
    return all_readable_tokens


def split(data: List[str], split_pat: str) -> List[List[str]]:
    split_pat = re.compile(split_pat)
    out: List[List[str]] = []
    group: List[str] = []
    for l in data:
        if split_pat.search(l) is not None and len(group) > 0:
            out.append(group)
            group = []
        else:
            group.append(l)
    if len(group) > 0:
        out.append(group)
    return out


def grep(data: List[str], pat: str, only_matched_text: bool = False) -> List[str]:
    pat = re.compile(pat)
    out = []
    for l in data:
        m = pat.search(l)
        if m is None:
            continue

        if only_matched_text:
            out.append(m.group())
        else:
            out.append(l)
    return out


def scan(data: List[str], start_pat: str = None, ending_pat: str = None, count: int = None, include_ending: bool = False) -> List[str]:
    assert count is None or isinstance(count, int) and count > 0
    start_pat = None if start_pat is None else re.compile(start_pat)
    ending_pat = None if ending_pat is None else re.compile(ending_pat)

    out = []
    i = 0
    while i < len(data):
        l = data[i]
        if start_pat is not None:
            m = start_pat.search(l)
            if m is None:
                i += 1
                continue

        # when either start_pat is None or start_pat is found, the start_pos is fixed at i
        if ending_pat is None and count is None:
            out.extend(data[i:])
            break

        if count is not None and count > 0:
            out.extend(data[i:i + count])
            i = i + count
            continue

        for j in range(i + 1, len(data)):
            ll = data[j]
            mm = ending_pat.search(ll)
            if mm is None:
                continue

            end_pos = j + 1 if include_ending else j
            out.extend(data[i:end_pos])
            # the ending is matched no matter included or not, so the next start pos will be the next line: j + 1
            i = j + 1
            break   # only the nearest ending pattern is required
        else:
            # if the ending pattern is never found after the text is exhausted,
            # all text from the start_pos until the end will be returned
            out.extend(data[i:])
            i = len(data)

    return out

