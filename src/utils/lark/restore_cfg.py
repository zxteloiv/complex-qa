from typing import List, Dict, Union, Generator, Tuple, Any, Optional
from collections import Counter, defaultdict
import lark
import utils.cfg as cfg
from .stat import RuleCollector
from .subtrees import EPS_TOK
import io
import os.path as osp
TREE, TOKEN = lark.Tree, lark.Token


def restore_grammar(rule_counter: Counter, rule_stat: RuleCollector) -> Tuple[cfg.T_CFG, dict]:
    terminal_vals = defaultdict(set)
    rule_lookup_table: Dict[int, TREE] = dict((r['id'], r['tree']) for r in rule_stat.rule_to_id.values())
    g: cfg.T_CFG = defaultdict(list)

    def _transform(children: List[Union[TREE, TOKEN]]) -> list:
        rhs = []
        for c in children:
            if isinstance(c, TREE):
                rhs.append(cfg.NonTerminal(c.data))
            else:
                rhs.append(cfg.Terminal(c.type))
                terminal_vals[c.type].add(c.value)
        return rhs

    for (nt, rid), count in rule_counter.items():
        if count == 0:
            continue
        tree = rule_lookup_table[rid]
        g[cfg.NonTerminal(tree.data)].append(_transform(tree.children))
    return g, terminal_vals


def restore_grammar_from_trees(trees: List[TREE]) -> Tuple[cfg.T_CFG, dict]:
    collector = RuleCollector()
    stats = collector.run_for_statistics(trees)
    counter = stats['rule_dist']
    return restore_grammar(counter, collector)


def export_grammar(g: cfg.T_CFG,
                   start: cfg.NonTerminal,
                   lex_in: Optional[str] = None,
                   export_terminal_vals: Optional[dict] = None,
                   excluded_terminals=None,
                   treat_terminals_as_categories: bool = False,
                   remove_eps: bool = True,
                   remove_useless: bool = True,
                   remove_unit_rules: bool = False,
                   ) -> str:
    if remove_eps:
        g = cfg.remove_eps_rules(g)
    if remove_useless:
        g = cfg.remove_useless_rules(g, start)
    if remove_unit_rules:
        g = cfg.remove_null_or_unit_rules(g)

    grammar_text = io.StringIO()
    if lex_in is not None and osp.isfile(lex_in):
        print(open(lex_in).read(), file=grammar_text)

    for lhs, rhs_list in g.items():
        print(f"{lhs.name}: " + ('\n' + (' ' * len(lhs.name)) + '| ').join(
            ' '.join(
                (
                    (f"P{t.name}" if t.name.startswith('_') else t.name)
                    if treat_terminals_as_categories else f'"{t.name}"'
                )
                if isinstance(t, cfg.Terminal) else t.name
                for t in rhs
            )
            for rhs in rhs_list
        ), file=grammar_text)

    if export_terminal_vals is not None:
        for tok, vals in export_terminal_vals.items():
            if tok not in excluded_terminals and tok != EPS_TOK.type:
                if tok.startswith('_'):
                    tok = 'P' + tok
                print(f"{tok}: " + ('\n' + (' ' * len(tok)) + '| ').join(
                    f'"{val}"i' if any(letter in val.lower() for letter in 'abcdefghijklmnopqrstuvwxyz')
                    else f'"{val}"'
                    for val in vals
                ), file=grammar_text)

    return grammar_text.getvalue()

