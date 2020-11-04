__doc__ = """
To compile a W3C EBNF grammar into a Lark grammar based on the Lark parser itself.
Main steps are as follows,

1. write a grammar in Lark that recognize the W3C ENBF grammar.
2. use lark to compile the grammar, and generate a parser capable of reading W3C ENBF.
3. use the parser to read the W3C EBNF grammar of SparQL
4. generate a lark-style grammar format.
"""
import lark

W3C_EBNF_Grammar = """
start: rule+
rule: symbol "::=" expr

symbol: identifier

?expr: atom | concat | alter
    
?token: identifier | literal

// expressions with parenthesis are unambiguous
optional: "(" expr ")" "?" | token "?"
kleene_star: "(" expr ")" "*" | token "*"
kleene_plus: "(" expr ")" "+" | token "+"
combine: "(" expr ")"
safe_concat: "(" expr+ ")"

?unambiguous_expr: optional | kleene_star | kleene_plus | combine | safe_concat

?atom: token | unambiguous_expr

concat: atom+

// concat has higher grammar priority than alter
?alter_piece: atom | concat
alter: _separated{alter_piece, "|"}

identifier: IDENTIFIER
literal: ESCAPED_STRING

IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
ESCAPED_STRING: "\"" /.*?/ "\"" | "'" /.*?/ "'"
C_COMMENT: "/*" /.*?/s "*/"

_separated{x, sep}: x (sep x)*  // Define a sequence of 'x sep x sep x ...'

%import common.WS
%ignore WS
%ignore C_COMMENT
"""

def main():
    from utils.root_finder import find_root
    import os.path
    grammar_file = os.path.join(find_root(), 'src', 'statics', 'grammar', 'sparql.ebnf')
    parser = lark.Lark(W3C_EBNF_Grammar)
    sparql_grammar = parser.parse(open(grammar_file).read())
    print(sparql_grammar.pretty())


def test():
    parser = lark.Lark(W3C_EBNF_Grammar)
    sparql_grammar_rule = """
    /* an example grammar rule */
    SelectClause ::= 'SELECT' ( 'DISTINCT' | 'REDUCED' )? ( ( Var | ( '(' Expression 'AS' Var ')' ) )+ | '*' )
    Prologue ::=  ( BaseDecl PrefixDecl )*
    Update1 ::=  Load | Clear | Drop | Add | Move | Copy | Create | InsertData | DeleteData | DeleteWhere | Modify
    Modify  ::=  ( 'WITH' iri )? ( DeleteClause InsertClause? | InsertClause ) UsingClause* 'WHERE' GroupGraphPattern
    GroupCondition	  ::=  	BuiltInCall | FunctionCall | '(' Expression ( 'AS' Var )? ')' | Var
    """
    tree = parser.parse(sparql_grammar_rule)
    print(tree.pretty())

if __name__ == '__main__':
    main()
    # test()