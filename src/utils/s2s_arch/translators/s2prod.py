from typing import Union, List, Tuple, Generator, Mapping, Optional
from trialbot.data.fields import SeqField
from trialbot.data.translator import FieldAwareTranslator
from trialbot.data.ns_vocabulary import START_SYMBOL, END_SYMBOL
from itertools import product
import lark
import torch
_Tree, _Token = lark.Tree, lark.Token


class TerminalRuleSeqField(SeqField):
    """Convert the sql into sequence of derivations guided by some grammar tree"""
    def __init__(self, keywords: dict = None, no_preterminals: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keywords = keywords or {}
        self.no_preterminals = no_preterminals

    def _get_rule_str(self, node: Union[_Tree, _Token]):
        if isinstance(node, _Tree):
            rule = [node.data]  # LHS
            children: List[Union[_Tree, _Token]] = node.children
            for tok in children:
                if isinstance(tok, _Tree):
                    rule.append(tok.data)
                elif self.no_preterminals:
                    rule.append(tok.value)
                elif tok.type in self.keywords:
                    rule.append(self.keywords[tok.type])
                else:
                    rule.append(tok.type)

            rule_str = ' '.join(rule)
        else:
            # the keywords terminals will not be here because they are ignored from the traverse_tree method.
            rule_str = node.type + ' ' + (node.value.lower() if self.lower_case else node.value)

        return rule_str

    def traverse_tree(self, tree: _Tree) -> Generator[Union[_Tree, _Token], None, None]:
        stack = [tree]
        while len(stack) > 0:
            node = stack.pop()
            yield node
            if isinstance(node, _Tree):
                for n in reversed(node.children):
                    if isinstance(n, _Token) and (self.no_preterminals or n.type in self.keywords):
                        continue
                    stack.append(n)

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        Tree, Token = lark.Tree, lark.Token
        tree: Tree = example.get(self.source_key)
        if tree is not None:
            for node in self.traverse_tree(tree):
                rule_str = self._get_rule_str(node)
                yield self.ns, rule_str

        if self.add_start_end_toks:
            yield from product([self.ns], [START_SYMBOL, END_SYMBOL])

    def to_tensor(self, example) -> Mapping[str, Optional['torch.Tensor']]:
        tree: _Tree = example.get(self.source_key)
        if tree is None:
            return {self.renamed_key: None}

        rule_id = [self.vocab.get_token_index(self._get_rule_str(n), self.ns) for n in self.traverse_tree(tree)]
        if self.max_seq_len > 0:
            rule_id = rule_id[:self.max_seq_len]

        if self.add_start_end_toks:
            start_id = self.vocab.get_token_index(START_SYMBOL, self.ns)
            end_id = self.vocab.get_token_index(END_SYMBOL, self.ns)
            rule_id = [start_id] + rule_id + [end_id]

        rule_seq_tensor = torch.tensor(rule_id)
        return {self.renamed_key: rule_seq_tensor}


class Seq2ProdTranslator(FieldAwareTranslator):
    def __init__(self,
                 source_field: str,
                 target_field: str,
                 source_max_token: int = 0,
                 target_max_token: int = 0,
                 use_lower_case: bool = True,
                 no_preterminals: bool = True,
                 ):
        super().__init__(field_list=[
            SeqField(source_key=source_field,
                     renamed_key='source_tokens',
                     namespace='source_tokens',
                     add_start_end_toks=False,
                     max_seq_len=source_max_token,
                     use_lower_case=use_lower_case),
            TerminalRuleSeqField(keywords=None,
                                 no_preterminals=no_preterminals,
                                 source_key=target_field,
                                 renamed_key='target_tokens',
                                 namespace='target_tokens',
                                 max_seq_len=target_max_token,
                                 add_start_end_toks=True,
                                 use_lower_case=use_lower_case,
                                 )

        ])
