from typing import List, Literal, TypeVar

PROMPT_CTX = TypeVar('PROMPT_CTX')


def build_ctx_with_exemplars(exemplars: List[dict],
                             source_field: str,
                             target_field: str,
                             use_ast: Literal['none', 'as_gen', 'as_brackets'] = 'none',
                             ast_field: str = None,
                             ) -> PROMPT_CTX:
    user_queries = [x.get(source_field) for x in exemplars]
    targets = [x.get(target_field) for x in exemplars]
    if use_ast == 'none':
        return list(zip(user_queries, targets))

    ast_field = ast_field or target_field + '_tree'
    ast2text_fn = ast2rules if use_ast == 'as_gen' else ast2brackets
    ast_texts = [ast2text_fn(x.get(ast_field)) for x in exemplars]
    return list(zip(user_queries, ast_texts, targets))


def ast2rules(ast):
    """transform an AST tree to texts"""
    from utils.lark.id_tree import build_from_lark_tree
    tree = build_from_lark_tree(ast)
    rules = []
    for subtree in tree.iter_subtrees_topdown():
        rules.append("{0} -> {1} ;".format(
            subtree.label, ' '.join(c.label for c in subtree.children)))
    prod_str = '\n'.join(rules)
    return prod_str


def ast2brackets(ast):
    from utils.lark.id_tree import build_from_lark_tree
    from utils.tree import InorderTraverse
    tree = build_from_lark_tree(ast)

    prod_str = ' '.join(
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
    )
    return prod_str


def build_prompt(x: str, ctx: PROMPT_CTX):
    use_syn = len(ctx[0]) > 2
    if use_syn:
        template = "Input: {0}\nGeneration: {1}\nOutput: {2}"
    else:
        template = "Input: {0}\nOutput: {1}"

    prompt = '\n'.join(template.format(*pair) for pair in ctx)
    prompt += f'\nInput: {x}\nOutput: '
    return prompt
