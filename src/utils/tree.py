from typing import TypeVar, Generator, Callable, List, Optional

T = TypeVar("T")

def preorder_traverse(root: T, children_fn: Optional[Callable[[T], List[T]]] = None) -> Generator[T, None, None]:
    if children_fn is None:
        children_fn = children_from_property

    yield root
    for c in children_fn(root):
        yield from preorder_traverse(c, children_fn)


def assign_node_id(tree, traversal):
    for i, x in enumerate(traversal(tree)):
        x.id = i
    return tree


def children_from_property(t):
    return t.children if hasattr(t, 'children') else []


def get_non_terminal_children(is_tree, children_fn):
    def non_terminal_children(t):
        return [c for c in children_fn(t) if is_tree(c)] if is_tree(t) else []
    return non_terminal_children


def non_terminal_children_from_property(t):
    is_tree = lambda x: hasattr(x, 'children')
    return get_non_terminal_children(is_tree, children_from_property)(t)


if __name__ == '__main__':
    import lark
    parser = lark.Lark("""
    start: ctok btok*
    ctok: atok*
    atok: /[aA]/
    btok: /[bB]/
    """, keep_all_tokens=True)
    s = "aaabb"
    tree = parser.parse(s)

    print(tree.pretty())

    for node in preorder_traverse(tree):
        if isinstance(node, lark.Tree):
            print(node.data)
        elif isinstance(node, lark.Token):
            print(node.type, node.value)
        else:
            raise NotImplementedError