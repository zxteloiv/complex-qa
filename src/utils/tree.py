from typing import TypeVar, Generator, Callable, List, Optional, Generic, Tuple, Union, Dict

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

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class InorderTraverse(Traversal):
    def __init__(self,
                 children_fn: T_CHILDREN_FN = None,
                 left_priority: bool = True,    # split children based on middle but left side first
                 output_parent: bool = False,
                 output_path: bool = False,
                 hooks: dict = None,
                 ):
        super().__init__(children_fn)
        self.left_prior = left_priority
        self.hooks: Dict[str, Callable] = hooks or dict()
        self.output_parent = output_parent
        self.output_path = output_path

    def __call__(self, root, /, hooks: dict = None):
        if hooks is not None:
            self.hooks.update(hooks)
        self._root = root
        yield from self._recursive(root, None, [])
        self._root = None

    def _recursive(self, node, parent, path):
        children = self.children_fn(node)
        length = len(children)

        if self.left_prior:
            # left-prior: [start:mid] [mid+1:len), mid >= len/2
            # caret: [0^], [0 ^ 1], [0 1 ^ 2], [0 1 ^ 2 3], [0 1 2 ^ 3 4]
            # mid:    1,    1,         2          2              3
            # len:    1,    2,         3          4              5
            mid = (length + 1) // 2

        else:
            # right prior:
            # caret: [^0], [0 ^ 1], [0 ^ 1 2], [0 1 ^ 2 3], [0 1 ^ 2 3 4]
            # mid:    0,    1,         1          2              2
            # len:    1,    2,         3          4              5
            mid = length // 2

        left_children = children[:mid] if length > 0 else []
        right_children = children[mid:] if length > 0 else []

        if 'pre_left_children' in self.hooks:
            yield from self.hooks['pre_left_children'](node, parent, path, self)
        for i, c in enumerate(left_children):
            yield from self._recursive(c, node, path + [i])

        if 'pre_visit' in self.hooks:
            yield from self.hooks['pre_visit'](node, parent, path, self)
        yield self.visit_node(node, parent, path)
        if 'post_visit' in self.hooks:
            yield from self.hooks['post_visit'](node, parent, path, self)

        for i, c in enumerate(right_children):
            yield from self._recursive(c, node, path + [i + len(left_children)])
        if 'post_right_children' in self.hooks:
            yield from self.hooks['post_right_children'](node, parent, path, self)

    def visit_node(self, node, parent, path):
        out = [node]
        if self.output_parent:
            out.append(parent)
        if self.output_path:
            out.append(path)
        if len(out) == 1:
            return out[0]
        else:
            return tuple(out)


class PreorderTraverse(Traversal):
    def __init__(self,
                 children_fn: T_CHILDREN_FN = None,
                 output_parent: bool = False,
                 output_path: bool = False,
                 hooks: dict = None,
                 ):
        super().__init__(children_fn)
        self.output_parent = output_parent
        self.output_path = output_path
        # there're several hooks to run
        self.hooks: Dict[str, Callable] = hooks or dict()

    def __call__(self,
                 root: T,
                 hooks: Dict[str, Callable] = None,
                 ) -> Generator[T, None, None]:
        if hooks is not None:
            self.hooks.update(hooks)
        self._root = root
        yield from self._recursive_call(root, None, [])
        self._root = None

    def _recursive_call(self, subtree: T, parent: T = None, path: List[int] = None):
        if 'pre_visit' in self.hooks:
            yield from self.hooks['pre_visit'](subtree, parent, path, self)
        yield self.visit_node(subtree, parent, path)
        if 'post_visit' in self.hooks:
            yield from self.hooks['post_visit'](subtree, parent, path, self)

        for i, c in enumerate(self.children_fn(subtree)):
            yield from self._recursive_call(c, subtree, path + [i])
        if 'post_children' in self.hooks:
            yield from self.hooks['post_children'](subtree, parent, path, self)

    def visit_node(self, node, parent, path) -> Union[T, Tuple[T]]:
        output = [node]
        if self.output_parent:
            output.append(parent)
        if self.output_path:
            output.append(path)
        if len(output) == 1:
            return output[0]
        else:
            return tuple(output)


class Tree:
    EPS_TOK: str = "%%EPS%%"

    def __init__(self, label: str, is_terminal: bool = False, children: list = None, payload=None):
        self.label = label
        self.node_id: int = 0

        # In general if is_terminal is set True, the children must be empty but we set them independently.
        # When the children list are empty, is_terminal can be set False because of the EPS rule.
        # In addition, however, we further differentiate a special situation where is_terminal is True and
        # the children list contains a single terminal. such as:
        #     Tree('NAME', is_terminal=True, children=[
        #         Tree('yuki kajiura', is_terminal=True)
        #     ])
        # This is an important use case in many CFG implementations and is usually called a preterminal.
        #
        # In summary,
        # common cases:
        #   nonterminals: is_terminal=False children=[A, B, ...]
        #      terminals: is_terminal=True  children=[]
        # special cases:
        #      EPS rules: is_terminal=False children=[], i.e., a nonterminal that emits nothing
        #   preterminals: is_terminal=True  children=[A-Common-Terminal], other cases are invalid
        #
        self.is_terminal = is_terminal
        self.children: List[Tree] = children or []
        self.payload = payload
        self.parent: Optional['Tree'] = None

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

    def build_parent_link(self):
        for c in self.children:
            c.parent = self
            c.build_parent_link()


if __name__ == '__main__':
    # test case 1: traversing a lark tree
    import sys
    import trialbot.utils.root_finder as rf
    sys.path.insert(0, rf.find_root('.SRC'))
    from utils.lark.id_tree import build_from_lark_tree
    import lark
    parser = lark.Lark("""
    start: ctok btok*
    ctok: atok*
    atok: /[aA]/
    btok: /[bB]/
    """, keep_all_tokens=True)
    s = "aaabb"
    tree = parser.parse(s)

    # for node in PreorderTraverse()(tree):
    #     if isinstance(node, lark.Tree):
    #         print(node.data)
    #     elif isinstance(node, lark.Token):
    #         print(node.type, node.value)
    #     else:
    #         raise NotImplementedError
    #
    # for node, node_parent in PreorderTraverse(output_parent=True)(tree):
    #     pref = node_parent.data + ' --->' if node_parent is not None else '<gen> --->'
    #     node: lark.Tree
    #
    #     if isinstance(node, lark.Tree):
    #         print(pref, node.data)
    #     elif isinstance(node, lark.Token):
    #         print(pref, node.type, node.value)
    #     else:
    #         raise NotImplementedError
    #
    # print('-----' * 10)
    tree = build_from_lark_tree(tree)
    #
    # def _post_children(node: Tree, parent: Optional[Tree], path, self: Traversal):
    #     RACT = ['REDUCE']
    #
    #     return [] if node.is_terminal else RACT
    #
    # for node in PreorderTraverse(hooks=dict(post_children=_post_children))(tree):
    #     if isinstance(node, str) and node == 'REDUCE':
    #         print(node)
    #     else:
    #         node: Tree
    #         if node.is_terminal:
    #             print("SHIFT:", node.label)
    #         else:
    #             print("NT:", node.label)

    print(' '.join(
        node if isinstance(node, str) else node.label   # no-qa
        for node in InorderTraverse()(tree, hooks={
            'pre_left_children': lambda n, parent, path, algo: "[",
            'post_right_children': lambda n, parent, path, algo: "]",
        })
    ))

    print(' '.join(
        node if isinstance(node, str) else node.label
        for node in InorderTraverse()(tree, hooks={
            'pre_left_children': lambda n, parent, path, algo: "[" if (
                    not n.is_terminal and len(algo.children_fn(n)) > 1
            ) else "",
            'post_right_children': lambda n, parent, path, algo: "]" if (
                    not n.is_terminal and len(algo.children_fn(n)) > 1
            ) else "",
        })
        if isinstance(node, str) or node.is_terminal
    ))
