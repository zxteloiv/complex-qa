from utils.sql_keywords import SQLITE_KEYWORDS, MYSQL_KEYWORDS, HANDCRAFTED_SQL_KEYWORDS
import utils.cfg as cfg


def get_export_conf(ds_name):
    if 'sqlite' in ds_name.lower():
        lex_file = 'SQLite.lark.lex-in'
        start = cfg.NonTerminal('parse')
        export_terminals = False
        excluded = list(SQLITE_KEYWORDS.keys())
    elif 'mysql' in ds_name.lower():
        lex_file = 'MySQL.lark.lex-in'
        start = cfg.NonTerminal('query')
        export_terminals = False
        excluded = list(MYSQL_KEYWORDS.keys())
    elif 'handcrafted' in ds_name.lower():
        lex_file = 'sql_handcrafted.lark.lex-in'
        start = cfg.NonTerminal('statement')
        export_terminals = True
        excluded = list(HANDCRAFTED_SQL_KEYWORDS.keys())
    elif 'cfq' in ds_name.lower():
        lex_file = 'sparql.lark.lex-in'
        start = cfg.NonTerminal('queryunit')
        export_terminals = True
        lex_terminals = "IRIREF PNAME_NS PNAME_LN BLANK_NODE_LABEL VAR1 VAR2 LANGTAG INTEGER DECIMAL DOUBLE " + \
                        "INTEGER_POSITIVE DECIMAL_POSITIVE DOUBLE_POSITIVE INTEGER_NEGATIVE DECIMAL_NEGATIVE " + \
                        "DOUBLE_NEGATIVE EXPONENT STRING_LITERAL1 STRING_LITERAL2 STRING_LITERAL_LONG1 " + \
                        "STRING_LITERAL_LONG2 ECHAR NIL WS ANON PN_CHARS_BASE PN_CHARS_U VARNAME PN_CHARS " + \
                        "PN_PREFIX PN_LOCAL PLX PERCENT HEX PN_LOCAL_ESC"
        excluded = lex_terminals.split()
    else:
        raise ValueError(f"dataset invalid: {ds_name}, failed to recognize the lexer and the start nonterminal")

    return lex_file, start, export_terminals, excluded