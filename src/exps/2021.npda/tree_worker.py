from typing import Callable, Optional, List, Union, Tuple
import lark
from enum import IntEnum, auto
import logging

TREE, TOKEN = lark.Tree, lark.Token
is_tree = lambda s: isinstance(s, TREE)


def enrich_tree(tree: TREE, parent=None) -> TREE:
    tree.parent = parent
    for c in tree.children:
        if is_tree(c):
            enrich_tree(c, parent=tree)
    return tree


class NodeAction(IntEnum):
    INTACT = auto()
    DEL_SELF = auto()
    ADD_PARENT = auto()
    L_ROTATE = auto()
    R_ROTATE = auto()
    L_DESC = auto()
    R_DESC = auto()


def modify_tree(tree: TREE, node_idx: int, action_id: int) -> bool:
    actions = [intact, del_self, add_parent, left_rotate, right_rotate, left_descent, right_descent]
    node, route = find_node_by_idx(tree, node_idx)

    foo: Callable[[TREE, List[int]], None] = actions[action_id]
    try:
        foo(node, route)
    except ValueError as e:
        logging.getLogger(__name__).warning(str(e))
        return False

    logging.getLogger(__name__).debug(f"successfully applied tree action {action_id}: {actions[action_id].__name__}")

    enrich_tree(tree)
    return True


def intact(node: TREE, route: List[int]) -> None:
    return


def find_node_by_idx(tree: TREE, idx: int) -> Tuple[TREE, List[int]]:
    node, node_id, route = None, 0, []
    op_stack: List[Tuple[TREE, List[int]]] = [(tree, [])]
    while len(op_stack) > 0:
        node, route = op_stack.pop()
        if node_id == idx:
            break

        for i, c in reversed(list(enumerate(node.children))):
            if is_tree(c):
                op_stack.append((c, route + [i]))
        node_id += 1
    else:
        raise ValueError('node of the specified index not found from the tree')

    return node, route


def get_parent(node: TREE) -> TREE:
    parent = getattr(node, 'parent', None)
    if parent is None:
        raise ValueError('root node must not be modified')
    return parent


def get_left_sibling(node: TREE, route: List[int]):
    parent = get_parent(node)
    pos = route[-1]
    if pos == 0:
        raise ValueError('left sibling not found')
    return parent.children[pos - 1]


def get_right_sibling(node: TREE, route: List[int]):
    parent = get_parent(node)
    pos = route[-1]
    if pos + 1 >= len(parent.children):
        raise ValueError('right sibling not found')
    return parent.children[pos + 1]


def get_leftmost_child(node: TREE):
    if not is_tree(node):
        raise ValueError('the node is not a non-terminal')
    if len(node.children) == 0:
        raise ValueError('children list is empty')

    return node.children[0]


def get_rightmost_child(node: TREE):
    get_leftmost_child(node)
    return node.children[-1]


def del_self(node: TREE, route: List[int]) -> None:
    parent = get_parent(node)
    pos = route[-1]
    parent.children[pos:pos+1] = node.children
    del node


def category_generation(node: TREE) -> str:
    max_val = 4
    val = min(len(node.children), max_val)
    return f"ANON_{val}"


def add_parent(node: TREE, route: List[int]) -> None:
    parent = get_parent(node)
    pos = route[-1]
    parent.children[pos] = TREE(category_generation(node), [node])


def left_rotate(node: TREE, route: List[int]) -> None:
    parent = get_parent(node)
    right = get_rightmost_child(node)
    pos = route[-1]
    parent.children[pos] = right
    node.children = node.children[:-1]
    right.children.insert(0, node)


def right_rotate(node: TREE, route: List[int]) -> None:
    parent = get_parent(node)
    left = get_leftmost_child(node)
    pos = route[-1]
    parent.children[pos] = left
    node.children = node.children[1:]
    left.children.append(node)


def left_descent(node: TREE, route: List[int]) -> None:
    parent = get_parent(node)
    pos = route[-1]
    left = get_left_sibling(node, route)
    parent.children = parent.children[:pos] + parent.children[pos + 1:]
    left.children.append(node)


def right_descent(node: TREE, route: List[int]) -> None:
    parent = get_parent(node)
    pos = route[-1]
    right = get_right_sibling(node, route)
    parent.children = parent.children[:pos] + parent.children[pos + 1:]
    right.children.insert(0, node)


if __name__ == '__main__':
    parser = lark.Lark("""
    start: ctok btok*
    ctok: atok ctok*
    atok: /[aA]/
    btok: /[bB]/
    """, keep_all_tokens=True)
    s = "aaaabb"
    tree = enrich_tree(parser.parse(s))
    print(tree.pretty())

    for idx in range(11):
        node, route = find_node_by_idx(tree, idx)
        print(route, node.data)

    node, route = find_node_by_idx(tree, 5)

    # del_self(node, route)
    # add_parent(node, route)
    # left_rotate(node, route)
    # right_rotate(node, route)
    # left_descent(node, route)
    # right_descent(node, route)

    enrich_tree(tree)

    print(tree.pretty())

    for idx in range(11):
        node, route = find_node_by_idx(tree, idx)
        print(route, node.data)
