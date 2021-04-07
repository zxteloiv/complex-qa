__doc__ = """
To compile a W3C EBNF grammar into a Lark grammar based on the Lark parser itself.
Main steps are as follows,

1. write a grammar in Lark that recognize the W3C ENBF grammar.
2. use lark to compile the grammar, and generate a parser capable of reading W3C ENBF.
3. use the parser to read the W3C EBNF grammar of SparQL
4. generate a lark-style grammar format.
"""
import sys
sys.path.insert(0, '../..')
import lark
import re
from utils.root_finder import find_root
from typing import List, Union
from os.path import join
from datetime import datetime as dt

W3C_EBNF_Grammar = join(find_root(), 'src', 'statics', 'grammar', 'w3c_ebnf.lark')

class EBNF2Lark(lark.Transformer):
    def concat(self, children):
        return " ".join(str(node) for node in children)

    def optional(self, children):
        return f"({children[0]})?"

    def kleene_star(self, children):
        return f"({children[0]})*"

    def kleene_plus(self, children):
        return f"({children[0]})+"

    def combine(self, children):
        return f"({children[0]})"

    def safe_concat(self, children):
        return f"({' '.join(children)})"

    def alter(self, children):
        return " | ".join(str(node) for node in children)

    def literal(self, children):
        literal = '"{0}"'.format(
            "".join(n.value.replace('\\', '\\\\').replace('"', '\\"') for n in children)
        )
        if re.search('[A-Z]', literal) is not None:
            literal = literal + 'i'
        return literal

    def rule(self, children):
        symbol, expr = children
        return f"{symbol}: {expr}"

    nonterminal_rule = rule
    terminal_rule = rule

    def nonterminal_symbol(self, children):
        return children[0].value.lower()

    def terminal_symbol(self, children):
        return children[0].value.upper()

    def inclusive_match(self, children):
        return "/[{chars}]/".format(chars="".join(str(n) for n in children))

    def exclusive_match(self, children):
        return "/[^{chars}]/".format(chars="".join(str(n) for n in children))

    def match_range(self, children):
        return "{0}-{1}".format(children[0], children[1])

    def codepoint(self, children):
        hexdigits = children[0].value
        length = len(hexdigits)
        if length <= 2:
            prefix, maxlength = "\\x", 2
        elif length <= 4:
            prefix, maxlength = "\\u", 4
        elif length <= 8:
            prefix, maxlength = "\\U", 8
        else:
            pass

        n =  "0" * (maxlength - length) + "".join(digit for digit in hexdigits)
        return prefix + n

    def string_cp(self, children):
        return "\"{0}\"".format(children[0])

from lark.visitors import Transformer_InPlace

from collections import defaultdict

class SymbolBasedExpansion(Transformer_InPlace):
    def __init__(self, terminal_name: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.terminal_name = terminal_name
        self.new_rules: List[Union[lark.Tree, lark.Token]] = []
        self.i = 0
        self.name_template = 'ext_{type}_{id}'

        self.ext_star_rules = defaultdict()
        self.ext_plus_rules = {}

    def _apply_new_name(self, type_):
        name = self.name_template.format(type=type_, id=self.i)
        self.i += 1
        if self.terminal_name:
            name = name.upper()
            new_t = lark.Tree('terminal_symbol', [lark.Token(name, name)])
        else:
            new_t = lark.Tree('nonterminal_symbol', [lark.Token(name, name)])

        return new_t

    @staticmethod
    def tree_hash(tree):
        return ' '.join(EBNF2Lark().transform(tree))

    def kleene_plus(self, children):
        exprkey = self.tree_hash(children[0])
        if exprkey in self.ext_plus_rules:
            return self.ext_plus_rules[exprkey]

        # if a kleene_star for the same expression is already used, new rule is not necessary
        # (expr)+ -> (({expr}) {star_name})
        # otherwise introducing a new name and a new rule
        # (expr)+ -> {new_name}
        # {new_name}: ({expr}) {new_name}?
        expr = children[0]
        star_rule = self.ext_star_rules.get(exprkey)
        if star_rule is not None:
            new_tree = lark.Tree('safe_concat', [
                lark.Tree('safe_concat', [expr]),
                star_rule
            ])
        else:
            new_tree = self._apply_new_name('plus')
            new_rule = lark.Tree('rule', [
                new_tree,
                lark.Tree('concat', [
                    lark.Tree('safe_concat', [expr]),
                    lark.Tree('optional', [new_tree])
                ])
            ])
            self.new_rules.append(new_rule)

        self.ext_plus_rules[exprkey] = new_tree
        return new_tree

    def kleene_star(self, children):
        exprkey = self.tree_hash(children[0])
        if exprkey in self.ext_star_rules:
            return self.ext_star_rules[exprkey]

        # if a kleene_plus of the same expr is already used, new rule is not necessary
        # (expr)* -> ({plus_name})?
        # otherwise introducing a new name and a new rule.
        # (expr)* -> {new_name}
        # {new_name}: ({expr} {new_name})?
        expr = children[0]
        plus_rule = self.ext_plus_rules.get(exprkey)
        if plus_rule is not None:
            new_tree = lark.Tree('optional', [plus_rule])
        else:
            new_tree = self._apply_new_name('star')
            new_rule = lark.Tree('rule', [
                new_tree,
                lark.Tree('optional', [
                    lark.Tree('concat', [
                        lark.Tree('safe_concat', [expr]),
                        new_tree
                    ])
                ])
            ])
            self.new_rules.append(new_rule)

        self.ext_star_rules[exprkey] = new_tree
        return new_tree

class EBNF2BNF(lark.Visitor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expansion = SymbolBasedExpansion()

    @property
    def new_rules(self):
        return self.expansion.new_rules

    # Expanding terminal rules is not required in lark, because the sparql parse tree
    # will not contain any derivation of terminals.
    # def terminal_rule(self, tree):
    #     self.expansion.terminal_name = True
    #     return self.expansion.transform(tree)

    def nonterminal_rule(self, tree):
        self.expansion.terminal_name = False
        return self.expansion.transform(tree)

def compile_file(in_file, out_file):
    text = open(in_file).read()
    out_text = compile(text)
    with open(out_file, 'w') as fout:
        name_suffix = dt.now().strftime('%s')
        print(f"// This file was automatically compiled"
              f" from the W3C standard of SparQL grammar at {dt.now().strftime('%Y-%m-%d %H:%M:%S')}"
              f"\n%import common.WS -> WS_{name_suffix}"
              f"\n%ignore WS_{name_suffix}\n", file=fout)

        print(out_text, file=fout)

def compile(in_text):
    parser = lark.Lark(open(W3C_EBNF_Grammar))
    sparql_ebnf_tree = parser.parse(in_text)

    # immediately processing the parse tree will yield a sparql.lark rather than .bnf.lark
    # lark_text = '\n'.join(EBNF2Lark().transform(sparql_ebnf_tree).children)

    bnf_converter = EBNF2BNF()
    bnf_tree = bnf_converter.visit(sparql_ebnf_tree)
    bnf_tree.children.extend(bnf_converter.new_rules)

    lark_text = '\n'.join(EBNF2Lark().transform(bnf_tree).children)
    return lark_text

def pretty_repr_tree(tree: lark.Tree, indent_str='    '):
    Tree = lark.Tree

    def _pretty(self: Tree, level, indent_str):
        if len(self.children) == 1 and not isinstance(self.children[0], Tree):
            return [indent_str * level, self._pretty_label(), '\t', '%r' % (self.children[0],), '\n']

        l = [indent_str * level, self._pretty_label(), '\n']
        for n in self.children:
            if isinstance(n, Tree):
                l += _pretty(n, level + 1, indent_str)
            else:
                l += [indent_str * (level + 1), '%r' % (n,), '\n']

        return l

    return ''.join(_pretty(tree, 0, indent_str))

def pretty_derivation_tree(tree: lark.Tree):
    rules = []
    for subtree in tree.iter_subtrees_topdown():
        lhs = subtree.data
        rhs = []
        for s in subtree.children:
            if isinstance(s, lark.Tree):
                rhs.append(s.data)
            elif isinstance(s, lark.Token):
                rhs.append(f"{s.type}({s.value})")
            else:
                raise ValueError(f"Symbol {s} not recognized")

        rhs = ' '.join(rhs)

        rules.append(f"{lhs} --> {rhs}")
    return rules

def test():
    sparql_lark = join(find_root(), 'src', 'statics', 'grammar', 'sparql_pattern.bnf.lark')
    sparql_parser = lark.Lark(open(sparql_lark), start="queryunit", keep_all_tokens=True,)

    # sparql = r"""
    #     SELECT count(*) WHERE {
    #     ?x0 ns:film.actor.film/ns:film.performance.character ns:m.011n3bs6 .
    #     ?x0 ns:film.editor.film ns:m.0_mhbxp .
    #     ?x0 ns:film.producer.film|ns:film.production_company.films ns:m.0_mhbxp .
    #     ?x0 ns:people.person.gender ns:m.02zsn
    #     }
    # """
    # sparql = r"""
    # SELECT DISTINCT ?x0 WHERE {
    # ?x0 a ns:film.editor .
    # ?x0 ns:people.person.spouse_s/ns:people.marriage.spouse|ns:fictional_universe.fictional_character.married_to/ns:fictional_universe.marriage_of_fictional_characters.spouses ns:m.079z_m .
    # ?x0 ns:people.person.spouse_s/ns:people.marriage.spouse|ns:fictional_universe.fictional_character.married_to/ns:fictional_universe.marriage_of_fictional_characters.spouses ns:m.0jwdyv3 .
    # FILTER ( ?x0 != ns:m.079z_m ) .
    # FILTER ( ?x0 != ns:m.0jwdyv3 )
    # }
    # """
    sparql = r"""
    SELECT DISTINCT ?x0 WHERE {
    ?x0 P0 M1 .
    ?x0 P0 M2 .
    ?x0 a M0 .
    FILTER ( ?x0 != M1 ) .
    FILTER ( ?x0 != M2 )
    }
    """
    tree = sparql_parser.parse(sparql)
    print(pretty_repr_tree(tree))
    print('\n'.join(pretty_derivation_tree(tree)))

    # sparql = r"""
    #     SELECT count(*) WHERE {
    #     ?x0 ns:film.actor.film/ns:film.performance.character M1 .
    #     ?x0 ns:film.editor.film M0 .
    #     ?x0 ns:film.producer.film|ns:film.production_company.films M0 .
    #     ?x0 ns:people.person.gender ns:m.02zsn
    #     }
    # """
    # print(pretty_repr_tree(sparql_parser.parse(sparql), indent_str='  '))
    # sparql = r"""
    #     SELECT count(*) WHERE {
    #     ?x0 P0 M0 ;
    #         P1 M0 .
    #     ?x0 P2 M1 .
    #     ?x0 P3 M2 .
    #     }
    # """
    # print(pretty_repr_tree(sparql_parser.parse(sparql), indent_str='  '))

def test2():
    ebnf = """
    start ::= a+ op? b* BB AA c JJ KK LL
    op ::= "+" | "-" | "*" | "/"
    a ::= AA
    b ::= BB
    c ::= a b
    AA ::= BB+
    BB ::= "b"
    JJ ::= BB*
    KK ::= AA*
    LL ::= AA+
    """
    parser = lark.Lark(open(W3C_EBNF_Grammar))
    ebnf_tree = parser.parse(ebnf)
    print(pretty_repr_tree(ebnf_tree))

    bnf_converter = EBNF2BNF()
    bnf_tree = bnf_converter.visit(ebnf_tree)
    bnf_tree.children.extend(bnf_converter.new_rules)
    print(pretty_repr_tree(bnf_tree))

    lark_text = EBNF2Lark().transform(bnf_tree)
    print("\n".join(lark_text.children))

def main(**kwargs):
    from argparse import ArgumentParser
    parser = ArgumentParser()
    grammar_path = kwargs.get('prefix', join(find_root(), 'src', 'statics', 'grammar'))
    parser.add_argument('--prefix', default=grammar_path, help='the path to grammar directory')
    parser.add_argument('--grammar', '-g', type=str, choices=['sparql', 'mysql', 'sqlite'], default=kwargs.get('grammar'))
    args = parser.parse_args()

    if args.grammar == 'sparql':
        grammar_file = join(args.prefix, 'sparql_pattern.ebnf')
        tgt_file = join(args.prefix, 'sparql_pattern.bnf.lark')
        compile_file(grammar_file, tgt_file)

    else:
        if args.grammar == 'sqlite':
            lex_file = join(args.prefix, 'SQLiteLexer.ebnf')
            parser_file = join(args.prefix, 'SQLiteParser.ebnf')
            tgt_file = join(args.prefix, 'SQLite.lark')
        elif args.grammar == 'mysql':
            lex_file = join(args.prefix, 'mysql-workbench', 'MySQLLexer.ebnf')
            parser_file = join(args.prefix, 'mysql-workbench', 'MySQLParser.ebnf')
            tgt_file = join(args.prefix, 'MySQL.lark')
        else:
            raise ValueError(f'grammar {args.grammar} not defined')

        text = open(lex_file).read() + '\n' + open(parser_file).read()
        lark_text = compile(text)
        print(lark_text)
        with open(tgt_file, 'w') as fout:
            name_suffix = dt.now().strftime('%s')
            print(f"// This file was automatically compiled"
                  f" from the W3C standard of SparQL grammar at {dt.now().strftime('%Y-%m-%d %H:%M:%S')}"
                  f"\n%import common.WS -> WS_{name_suffix}"
                  f"\n%ignore WS_{name_suffix}\n", file=fout)
            print(lark_text, file=fout)
            print("\n%ignore WS\n", file=fout)

if __name__ == '__main__':
    main(grammar="mysql")
    # test()
    # test2()
