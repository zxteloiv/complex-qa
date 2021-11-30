from typing import Callable, Optional, List, Union, Tuple
from enum import IntEnum, auto
import logging
from utils.tree import Tree, PreorderTraverse


def build_parent_link(t: Tree, parent=None) -> Tree:
    t.parent = parent
    for c in t.children:
        build_parent_link(c, parent=t)
    return t


def modify_tree(t: Tree, node_idx: int, action_id: int) -> bool:
    actions = [del_self, add_parent, left_rotate, right_rotate, left_descent, right_descent, left_ascent, right_ascent]
    t = build_parent_link(t)

    try:
        node, route = find_node_by_idx(t, node_idx)
        if node.is_terminal and action_id not in (4, 5, 6, 7):
            raise ValueError(f'the terminal node {node_idx}:{node.label} shall not be modified')
        foo: Callable[[Tree, List[int]], None] = actions[action_id]
        foo(node, route)
    except ValueError as e:
        logging.getLogger(__name__).warning(str(e))
        return False

    logging.getLogger(__name__).debug(f"successfully applied to {node.label} action {actions[action_id].__name__}")

    build_parent_link(t)
    return True


def find_node_by_idx(tree: Tree, idx: int) -> Tuple[Tree, List[int]]:
    id_tree = tree.assign_node_id(PreorderTraverse())
    id_tree = build_parent_link(id_tree)

    for node, route in PreorderTraverse(output_path=True)(id_tree):
        if node.node_id == idx:
            return node, route

    raise ValueError('node of the specified index not found from the tree')


def get_parent(node: Tree) -> Tree:
    parent = getattr(node, 'parent', None)
    if parent is None:
        raise ValueError('root node must not be modified')
    return parent


def get_left_sibling(node: Tree, route: List[int]):
    parent = get_parent(node)
    pos = route[-1]
    if pos == 0:
        raise ValueError('left sibling not found')
    return parent.children[pos - 1]


def get_right_sibling(node: Tree, route: List[int]):
    parent = get_parent(node)
    pos = route[-1]
    if pos + 1 >= len(parent.children):
        raise ValueError('right sibling not found')
    return parent.children[pos + 1]


def get_leftmost_child(node: Tree):
    if node.is_terminal:
        raise ValueError('the node is a terminal')
    if len(node.children) == 0:
        raise ValueError('children list is empty')

    return node.children[0]


def get_rightmost_child(node: Tree):
    get_leftmost_child(node)
    return node.children[-1]


def del_self(node: Tree, route: List[int]) -> None:
    parent = get_parent(node)
    pos = route[-1]
    parent.children[pos:pos+1] = node.children
    del node


def category_generation(node: Tree) -> str:
    max_val = 4
    val = min(len(node.children), max_val)
    return f"anon_nt_{val}"


def add_parent(node: Tree, route: List[int]) -> None:
    parent = get_parent(node)
    pos = route[-1]
    parent.children[pos] = Tree(category_generation(node), is_terminal=False, children=[node])


def left_rotate(node: Tree, route: List[int]) -> None:
    parent = get_parent(node)
    right = get_rightmost_child(node)
    if right.is_terminal:
        raise ValueError("Failed to rotate on a terminal pivot")
    pos = route[-1]
    parent.children[pos] = right
    node.children = node.children[:-1]
    right.children.insert(0, node)


def right_rotate(node: Tree, route: List[int]) -> None:
    parent = get_parent(node)
    left = get_leftmost_child(node)
    if left.is_terminal:
        raise ValueError("Failed to rotate on a terminal pivot")
    pos = route[-1]
    parent.children[pos] = left
    node.children = node.children[1:]
    left.children.append(node)


def left_descent(node: Tree, route: List[int]) -> None:
    parent = get_parent(node)
    pos = route[-1]
    left = get_left_sibling(node, route)
    if left.is_terminal:
        raise ValueError("Failed to descend under a terminal sibling.")
    parent.children = parent.children[:pos] + parent.children[pos + 1:]
    left.children.append(node)


def right_descent(node: Tree, route: List[int]) -> None:
    parent = get_parent(node)
    pos = route[-1]
    right = get_right_sibling(node, route)
    if right.is_terminal:
        raise ValueError("Failed to descend under a terminal sibling.")
    parent.children = parent.children[:pos] + parent.children[pos + 1:]
    right.children.insert(0, node)


def left_ascent(node: Tree, route: List[int]) -> None:
    parent = get_parent(node)
    grandparent = get_parent(parent)
    pos = route[-1]
    if pos > 0:
        raise ValueError("Failed to left ascend a non-leftmost node")
    parent_pos = route[-2]

    parent.children = parent.children[pos + 1:]
    grandparent.children.insert(parent_pos, node)


def right_ascent(node: Tree, route: List[int]) -> None:
    parent = get_parent(node)
    grandparent = get_parent(parent)
    pos = route[-1]
    if pos < len(parent.children) - 1:
        raise ValueError("Failed to right ascend a non-rightmost node")

    parent_pos = route[-2]

    parent.children = parent.children[:-1]
    grandparent.children.insert(parent_pos + 1, node)


if __name__ == '__main__':
    import lark, sys
    from utils.lark.id_tree import build_from_lark_tree
    parser = lark.Lark("""
    start: ctok btok*
    ctok: atok ctok*
    atok: /[aA]/
    btok: /[bB]/
    """, keep_all_tokens=True)
    s = "aaaabb"
    tree = build_from_lark_tree(parser.parse(s), add_eps_nodes=True).assign_node_id(PreorderTraverse())
    print(tree)
    for n in PreorderTraverse()(tree):
        print(n.node_id, ":", n.label, '-->', ' '.join(c.immediate_str() for c in n.children))

    for n, path in PreorderTraverse(output_path=True)(tree):
        print('    ' * len(path), (n.node_id, n.label))

    # del_self(node, route)
    # add_parent(node, route)
    # left_rotate(node, route)
    # right_rotate(node, route)
    # left_descent(node, route)
    # right_descent(node, route)
    left_ascent(*find_node_by_idx(tree, 3))
    # right_ascent(*find_node_by_idx(tree, 7))

    build_parent_link(tree)
    print(tree)

    for n, path in PreorderTraverse(output_path=True)(tree):
        print('    ' * len(path), (n.node_id, n.label))

    node, route = find_node_by_idx(tree, 13)
    print(node.immediate_str(), route)

