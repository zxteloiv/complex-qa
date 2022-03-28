# hardcode the conversion between a token (either a category or a terminal) and the id
from typing import Union
from trialbot.data import NSVocabulary

from utils.tree import Tree


# type conventions within this utility:
#
# token_str: str, a string representing the token;
# token: Tree, a node on the parse tree, either a terminal or non-terminal;
# action_id: int
#
# the pipeline is like:
#    token ---> token_str <---> action_id
#


def get_token_str(tok: Union[Tree, str]):
    if isinstance(tok, str):
        return tok
    if tok.is_terminal:
        return tok.label
    else:
        return 'NT:' + tok.label


# reduce token is not encoded in a tree explicitly, and thus we need to appended it via a hook
# and build a tree object for it.
def get_reduce_token() -> Tree: return Tree('RNNG:REDUCE', is_terminal=True)


def is_rnng_action(tok: Union[Tree, str]):
    if isinstance(tok, Tree):
        return tok.is_terminal and tok.label.startswith('RNNG:')
    elif isinstance(tok, str):
        return tok.startswith('RNNG:')
    else:
        raise TypeError(f'unknown value type {type(tok)}')


def is_nt_action(tok: Union[Tree, str]):
    if isinstance(tok, Tree):
        return not tok.is_terminal
    elif isinstance(tok, str):
        return tok.startswith('NT:')
    else:
        raise TypeError(f'unknown value type {type(tok)}')


def is_gen_action(tok: Union[Tree, str]):
    if isinstance(tok, Tree):
        return tok.is_terminal and not is_rnng_action(tok)
    elif isinstance(tok, str):
        return not tok.startswith('NT:') and not tok.startswith('RNNG:')
    else:
        raise TypeError(f'unknown value type {type(tok)}')


# 0    1    2       3  ...  boundary ...
# PAD  OOV  REDUCE  NT ...  SHIFT    ...


def token_to_id(tok: Union[Tree, str], vocab: NSVocabulary, namespaces: tuple):
    ns_rnng, ns_nt, ns_term = namespaces
    rnng_sz = vocab.get_vocab_size(ns_rnng)
    nt_sz = vocab.get_vocab_size(ns_nt)

    if is_rnng_action(tok):
        return vocab.get_token_index(get_token_str(tok), ns_rnng)
    elif is_nt_action(tok):
        return rnng_sz + vocab.get_token_index(get_token_str(tok), ns_nt)
    else:
        return rnng_sz + nt_sz + vocab.get_token_index(get_token_str(tok), ns_term)


def id_to_token_str(action_id: int, vocab: NSVocabulary, namespaces: tuple) -> str:
    ns_rnng, ns_nt, ns_term = namespaces
    rnng_sz = vocab.get_vocab_size(ns_rnng)
    nt_sz = vocab.get_vocab_size(ns_nt)
    t_sz = vocab.get_vocab_size(ns_term)

    if action_id < rnng_sz:
        return vocab.get_token_from_index(action_id, ns_rnng)
    elif action_id < rnng_sz + nt_sz:
        return vocab.get_token_from_index(action_id - rnng_sz, ns_nt)
    elif action_id < rnng_sz + nt_sz + t_sz:
        return vocab.get_token_from_index(action_id - rnng_sz - nt_sz, ns_term)
    else:
        raise ValueError(f'invalid action id {action_id} found')


def get_terminal_boundary(vocab: NSVocabulary, namespaces: tuple) -> int:
    ns_rnng, ns_nt, ns_term = namespaces
    rnng_sz = vocab.get_vocab_size(ns_rnng)
    nt_sz = vocab.get_vocab_size(ns_nt)
    return rnng_sz + nt_sz


def get_target_num_embeddings(vocab: NSVocabulary, namespaces: tuple) -> int:
    ns_rnng, ns_nt, ns_term = namespaces
    rnng_sz = vocab.get_vocab_size(ns_rnng)
    nt_sz = vocab.get_vocab_size(ns_nt)
    term_sz = vocab.get_vocab_size(ns_term)
    return rnng_sz + nt_sz + term_sz
