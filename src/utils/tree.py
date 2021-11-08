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
    def __init__(self, children_fn: T_CHILDREN_FN = None):
        self.children_fn: T_CHILDREN_FN = children_fn or children_by_property

    def __call__(self, root: T):
        raise NotImplementedError


class PreorderTraverse(Traversal):
    def __init__(self, children_fn: T_CHILDREN_FN = None, output_parent: bool = False, output_path: bool = False):
        super().__init__(children_fn)
        self.output_parent = output_parent
        self.output_path = output_path

    def __call__(self, root: T):
        yield from self._recursive_call(root, path=[])

    def _recursive_call(self, root: T, parent: T = None, path: List[int] = None):
        output = [root]
        if self.output_parent:
            output.append(parent)
        if self.output_path:
            output.append(path)
        if len(output) == 1:
            yield output[0]
        else:
            yield tuple(output)

        for i, c in enumerate(self.children_fn(root)):
            yield from self._recursive_call(c, root if self.output_parent else None, path + [i] if self.output_path else None)


class Tree(Generic[T]):
    EPS_TOK: str = "%%EPS%%"

    def __init__(self, label: str, is_terminal: bool = False, children: list = None, payload = None):
        self.label = label
        self.node_id: int = 0

        # Generally if is_terminal is set True, the children must be empty but we set them independently.
        # When the children list are empty, is_terminal can be set False because of the EPS rule.
        # In addition, however, we further differentiate a special situation where is_terminal is True and
        # the children list contains a single terminal. such as:
        #     Tree('NAME', is_terminal=True, children=[
        #         Tree('yuki kajiura', is_terminal=True)
        #     ])
        # This is an important use case in the parse trees of many CFG grammars and parsers.
        self.is_terminal = is_terminal
        self.children: List[Tree] = children or []
        self.payload = payload

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

    def iter_subtrees_topdown(self) -> Generator['Tree', None, None]:
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

    for node, node_parent in PreorderTraverse(output_parent=True)(tree):
        pref = node_parent.data + ' --->' if node_parent is not None else '<gen> --->'

        if isinstance(node, lark.Tree):
            print(pref, node.data)
        elif isinstance(node, lark.Token):
            print(pref, node.type, node.value)
        else:
            raise NotImplementedError
