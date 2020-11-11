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
import os.path

W3C_EBNF_Grammar = open(os.path.join(find_root(), 'src', 'statics', 'grammar', 'w3c_ebnf.lark'))

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

def compile():
    grammar_file = os.path.join(find_root(), 'src', 'statics', 'grammar', 'sparql.ebnf')
    parser = lark.Lark(W3C_EBNF_Grammar)
    sparql_grammar = parser.parse(open(grammar_file).read())

    lark_tree = EBNF2Lark().transform(sparql_grammar)
    sparql_grammar_in_lark = "\n".join(lark_tree.children)
    tgt_file = os.path.join(find_root(), 'src', 'statics', 'grammar', 'sparql.lark')

    from datetime import datetime as dt
    with open(tgt_file, 'w') as fout:
        name_suffix = dt.now().strftime('%s')
        print(f"// This file was automatically compiled"
              f" from the W3C standard of SparQL grammar at {dt.now().strftime('%Y-%m-%d %H:%M:%S')}"
              f"\n%import common.WS -> WS_{name_suffix}"
              f"\n%ignore WS_{name_suffix}\n", file=fout)

        print(sparql_grammar_in_lark, file=fout)

def data_expansion():
    """
    Load the sparql lark grammar (and lark will expand it by default).
    Dump the processed data for grammar warmup.
    """
    sparql_lark = os.path.join(find_root(), 'src', 'statics', 'grammar', 'sparql.lark')
    sparql_lark_expanded = os.path.join(find_root(), 'src', 'statics', 'grammar', 'sparql_expand.lark')
    parser = lark.Lark(open(sparql_lark), start="queryunit", keep_all_tokens=True)
    print(parser.rules)

    with open(sparql_lark_expanded, 'w') as fout:
        for rule in parser.rules:
            pass
        pass



def test():
    sparql_lark = os.path.join(find_root(), 'src', 'statics', 'grammar', 'sparql.lark')
    sparql_parser = lark.Lark(open(sparql_lark), start="queryunit", keep_all_tokens=True)
    sparql = r"""
        SELECT count(*) WHERE {
        ?x0 ns:film.actor.film/ns:film.performance.character ns:m.011n3bs6 .
        ?x0 ns:film.editor.film ns:m.0_mhbxp .
        ?x0 ns:film.producer.film|ns:film.production_company.films ns:m.0_mhbxp .
        ?x0 ns:people.person.gender ns:m.02zsn
        }
    """
    print(sparql_parser.parse(sparql).pretty())
    sparql = r"""
        SELECT count(*) WHERE {
        ?x0 :P0 :M0 .
        ?x0 :P1 :M0 .
        ?x0 :P2 :M1 .
        ?x0 :P3 :M2
        }
    """
    print(sparql_parser.parse(sparql).pretty())
    pass

if __name__ == '__main__':
    # compile()
    # data_expansion()
    test()
