from lark import Lark


def main():
    from utils.lark.id_tree import build_from_lark_tree
    from utils.tree import InOrderTraverse, PostOrderTraverse, PreOrderTraverse, Tree
    pass
    from shujuji.smcalflow_cs import smc_by_num, GRAMMAR_FILE
    train, dev, test = smc_by_num(128)
    parser = Lark(open(GRAMMAR_FILE), keep_all_tokens=True)
    xs = train[128:138]
    tree = build_from_lark_tree(parser.parse(xs[0]['plan']))

    tree2 = Tree(label='s', children=[
        Tree(label='A', is_terminal=True),
        Tree(label='bcd', children=[
            Tree(label='B', is_terminal=True),
            Tree(label='cd', children=[
                Tree(label='C', is_terminal=True),
                Tree(label='D', is_terminal=True),
            ]),
        ]),
        Tree(label='E', is_terminal=True),
        Tree(label='fg', children=[
            Tree(label='F', is_terminal=True),
            Tree(label='g', children=[
                Tree(label='G', is_terminal=True),
            ]),
        ]),
    ])
    foo(tree2)


def foo(tree):
    from utils.tree import InOrderTraverse, PostOrderTraverse, PreOrderTraverse, Tree
    # set payload for node offset
    running_prefix = 0
    for node in PostOrderTraverse()(tree):
        if node.is_terminal:
            node.payload = (running_prefix, running_prefix + len(node.label))
            running_prefix += 1 + len(node.label)  # the space sep
        else:
            # use the left-most to right-most
            node.payload = (node.children[0].payload[0], node.children[-1].payload[-1])

    text = ' '.join(node.label for node in PostOrderTraverse()(tree) if node.is_terminal)
    print('length:', len(text))
    print(text)
    print('====' * 30)
    for node, path in PreOrderTraverse(output_path=True)(tree):
        start, end = node.payload
        if node.is_terminal:
            indent = '__' * len(path)
            print(indent + node.label, f'({start},{end})')
            continue
        else:
            indent = '__' * len(path)
            print(indent + node.label, f'({start},{end})', '--->', text[slice(*node.payload)])


if __name__ == '__main__':
    import sys
    from trialbot.utils.root_finder import find_root
    sys.path.insert(0, find_root('.SRC'))
    main()
