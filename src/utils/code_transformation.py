import ast, astor

class CodeTransform:
    @staticmethod
    def dump_python_ast_tree(s):
        return s.replace('(', '{').replace(')', '}')

    @staticmethod
    def dump_python(code, special=ast.AST):
        node = ast.parse(code)
        def dump(node, name=None):
            name = name or ''
            values = list(astor.iter_node(node))
            if isinstance(node, list):
                prefix, suffix = '{%s' % name, '}'
            elif values:
                prefix, suffix = '{%s' % type(node).__name__, '}'
            elif isinstance(node, special):
                prefix, suffix = '{%s' % name + type(node).__name__, '}'
            else:
                return '{%s}' % type(node).__name__
            node = [dump(a, b) for a, b in values if b != 'ctx']
            return '%s %s %s' % (prefix, ' '.join(node), suffix)
        return dump(node)

    @staticmethod
    def dump_lambda(s):
        s = s.strip()
        if not s.startswith('('):
            s = '( ' + s + ' )'

        tokens = s.split(' ')
        for i, tok in enumerate(tokens[1:], start=1):
            if tokens[i - 1] != '(' and tok not in ('(', ')'):
                tokens[i] = '(%s)' % tok

        out = ' '.join(tokens).replace('(', '{').replace(')', '}')
        return out


