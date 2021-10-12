from typing import Union
import lark
TREE, TOKEN = lark.Tree, lark.Token

def compact_hash(t: Union[TREE, TOKEN]):
    if isinstance(t, TOKEN):
        return t.type

    lhs = t.data
    # only the categories of terminals is considered, the values are not used
    rhs = ' '.join(compact_hash(c) for c in t.children)
    return f"{lhs}: ({rhs})"

