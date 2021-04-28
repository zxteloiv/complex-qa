from typing import List, Mapping, Generator, Tuple, Optional, Any, Literal, Iterable, Union, DefaultDict
import torch
from .field import Field
from .seq_field import SeqField
import lark
from trialbot.data import START_SYMBOL, END_SYMBOL
_Tree, _Token = lark.Tree, lark.Token
from itertools import product
from torch.nn.utils.rnn import pad_sequence

class TerminalRuleSeqField(SeqField):
    def _get_rule_str(self, node: Union[_Tree, _Token]):
        if isinstance(node, _Tree):
            rule = [node.data]  # LHS
            children: List[Union[_Tree, _Token]] = node.children
            for tok in children:
                rule.append(tok.data if isinstance(tok, _Tree) else tok.type)
            rule_str = ' '.join(rule)
        else:
            # lower case only applies to the node value
            rule_str = node.type + ' ' + (node.value.lower() if self.lower_case else node.value)

        return rule_str

    @classmethod
    def traverse_tree(cls, tree: _Tree) -> Generator[Union[_Tree, _Token], None, None]:
        stack = [tree]
        while len(stack) > 0:
            node = stack.pop()
            yield node
            if isinstance(node, _Tree):
                for n in reversed(node.children):
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

    def to_tensor(self, example) -> Mapping[str, Optional[torch.Tensor]]:
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


