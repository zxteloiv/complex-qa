from typing import TypeVar, Generator, Callable, List, Optional, Generic

T = TypeVar('T')
T_CHILDREN_FN = Callable[[T], List[T]]


# by default, the children_by_property could be applicable to lark.Tree, so could the Traversal objects.
def children_by_property(t: T) -> List[T]:
    return t.children if hasattr(t, 'children') else []


def get_filtered_children_fn(pred, fn: T_CHILDREN_FN) -> T_CHILDREN_FN:
    def filtered_children(t: T):
        return [c for c in fn(t) if pred(c)]
    return filtered_children


class Traversal:
    def __init__(self, children_fn: Callable[[T], List[T]] = None):
        self.children_fn: Callable[[T], List[T]] = children_fn or children_by_property

    def __call__(self, root: T) -> List[T]:
        raise NotImplementedError


class PreorderTraverse(Traversal):
    def __init__(self, children_fn=children_by_property):
        super().__init__(children_fn)

    def __call__(self, root):
        yield root
        for c in self.children_fn(root):
            yield from self(c)


class Tree(Generic[T]):
    EPS_TOK: str = "%%EPS%%"

    def __init__(self, label: str, is_terminal: bool, children: list = None):
        self.label = label
        self.node_id: int = 0

        self.is_terminal = is_terminal
        self.children: List[Tree] = children or []
        if is_terminal:  # forced to be empty for a terminal
            self.children: List[Tree] = []

    def __str__(self):
        if len(self.children) == 0:
            return self.label
        return f"{self.label} ({' '.join(str(c) for c in self.children)})"

    def immediate_str(self):
        if len(self.children) == 0:
            return self.label
        return f"{self.label} ({' '.join(c.label for c in self.children)})"

    @property
    def has_empty_rhs(self):
        return len(self.children) == 0

    def iter_subtrees_topdown(self):
        """mimic the behavior of a lark.Tree"""
        yield from PreorderTraverse(Tree.get_nt_children_fn())(self)

    @classmethod
    def get_nt_children_fn(cls):
        nt_pred = lambda c: not c.is_terminal
        return get_filtered_children_fn(nt_pred, children_by_property)

    def assign_node_id(self, traversal: Traversal):
        for i, x in enumerate(traversal(self)):
            x.node_id = i
        return self


if __name__ == '__main__':
    # test case 1: traversing a lark tree
    import lark
    parser = lark.Lark("""
    start: ctok btok*
    ctok: atok*
    atok: /[aA]/
    btok: /[bB]/
    """, keep_all_tokens=True)
    s = "aaabb"
    tree = parser.parse(s)

    for node in PreorderTraverse()(tree):
        if isinstance(node, lark.Tree):
            print(node.data)
        elif isinstance(node, lark.Token):
            print(node.type, node.value)
        else:
            raise NotImplementedError