from typing import List, Mapping, Generator, Tuple, Optional, Any, Literal, Iterable, Union, DefaultDict
import torch
from .field import Field
from .seq_field import SeqField
import lark
from trialbot.data import START_SYMBOL, END_SYMBOL
_Tree, _Token = lark.Tree, lark.Token
from itertools import product
from torch.nn.utils.rnn import pad_sequence
import nltk
import re

class ProcessedSentField(SeqField):
    def get_sent(self, example):
        sent = super().get_sent(example)
        return self.process_sentence(sent)

    """
    Code adapted from https://github.com/jkkummerfeld/text2sql-data/
    """
    @classmethod
    def process_sentence(cls, sentence):
        sentence = cls.dept_num_spacing(sentence)
        sentence = cls.am_and_pm(sentence)
        sentence = cls.standardize_word_forms(sentence)
        sentence = cls.n_credit(sentence)
        sentence = sentence.strip()
        sentence = sentence.replace('``', '"')
        return sentence

    @classmethod
    def n_credit(cls, sentence):
        """
        If a phrase in the form "X credit" appears in the sentence, and
        X is a number, add a hyphen to "credit".
        If "X-credit" appears, split it into "X" and "-credit".
        Do not hyphenate "X credits" or "Y credit" where Y is not a number.
        Run this after the tokenized text has been joined.
        >>> n_credit('I need a 3 credit course .')
        'I need a 3 -credit course .'
        >>> n_credit('I need a 3-credit course .')
        'I need a 3 -credit course .'
        >>> n_credit('I need a course worth 3 credits .')
        'I need a course worth 3 credits .'
        >>> n_credit('Can I get credit ?')
        'Can I get credit ?'
        """
        pattern = r"(?P<number>\d)+[- ]credit\s"
        repl = r"\g<number> -credit "
        return re.sub(pattern, repl, sentence)

    @classmethod
    def dept_num_spacing(cls, sentence):
        """
        Given a sentence with a department abbreviation followed by a course number,
        ensure that there's a space between the abbreviation and number.
        An all-caps string of exactly 4 letters or the string "eecs" is considered
        a department if it is followed immediately by a 3-digit number.
        Run this before tokenizing.
        >>> dept_num_spacing("EECS280")
        'EECS 280'
        >>> dept_num_spacing("MATH417")
        'MATH 417'
        >>> dept_num_spacing("eecs280")
        'eecs 280'
        >>> dept_num_spacing("gEECS365")
        'gEECS365'
        >>> dept_num_spacing("EECS280 and MATH417")
        'EECS 280 and MATH 417'
        """
        pattern = r"(?P<dept>^[A-Z]{4}|\s[A-Z]{4}|eecs)(?P<number>\d{3})"
        repl = r"\g<dept> \g<number>"
        return re.sub(pattern, repl, sentence)

    @classmethod
    def am_and_pm(cls, sentence):
        """
        Standardize variations as "A.M." or "P.M." iff they appear after a time.
        >>> am_and_pm("at twelve pm")
        'at twelve P.M.'
        >>> am_and_pm("at 12 pm")
        'at 12 P.M.'
        >>> am_and_pm("I am on a boat")
        'I am on a boat'
        >>> am_and_pm("9 am classes")
        '9 A.M. classes'
        >>> am_and_pm("9 AM classes")
        '9 A.M. classes'
        >>> am_and_pm("9:30 AM classes")
        '9:30 A.M. classes'
        >>> am_and_pm("9AM classes")
        '9 A.M. classes'
        >>> am_and_pm("is 280 among")
        'is 280 among'
        """
        number_pattern = r"(?P<time>(^|\s|[A-Za-z:])\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|thirty|forty-?five|o'clock) ?(?P<meridian>"
        am_pattern = number_pattern + r"am|AM|a\.m\.|A\.M\.)"
        pm_pattern = number_pattern + r"pm|PM|p\.m\.|P\.M\.)"

        am_repl = r"\g<time> A.M."
        pm_repl = r"\g<time> P.M."

        sentence = re.sub(am_pattern, am_repl, sentence)
        return re.sub(pm_pattern, pm_repl, sentence)

    @classmethod
    def standardize_word_forms(cls, sentence):
        """
        Replace words with a standardized version.
        >>> standardize_word_forms("Does dr smith teach any courses worth four credits?")
        'Does Dr. smith teach any courses worth 4 credits ?'
        """
        # TODO: make a JSON with the word-forms dict
        corrections = {"one": "1",
                       "two": "2",
                       "three": "3",
                       "four": "4",
                       "five": "5",
                       "six": "6",
                       "seven": "7",
                       "eight": "8",
                       "nine": "9",
                       "ten": "10",
                       "eleven": "11",
                       "twelve": "12",
                       "dr": "Dr.",
                       "Dr": "Dr.",
                       "dr.": "Dr.",
                       "Prof": "Professor",
                       "Professor": "Professor",
                       "prof.": "Professor",
                       "eecs": "EECS"
                       }
        tokens = nltk.word_tokenize(sentence)
        correct_tokens = []
        for word in tokens:
            if word in corrections:
                correct_tokens.append(corrections[word])
            else:
                correct_tokens.append(word)
        return " ".join(correct_tokens)

class TerminalRuleSeqField(SeqField):
    def __init__(self, keywords: dict = None, no_terminal_rule: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keywords = keywords or {}
        self.no_terminal_rules = no_terminal_rule

    def _get_rule_str(self, node: Union[_Tree, _Token]):
        if isinstance(node, _Tree):
            rule = [node.data]  # LHS
            children: List[Union[_Tree, _Token]] = node.children
            for tok in children:
                if isinstance(tok, _Tree):
                    rule.append(tok.data)
                elif self.no_terminal_rules:
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
                    if isinstance(n, _Token) and (self.no_terminal_rules or n.type in self.keywords):
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

class RuleSymbolSeqField(TerminalRuleSeqField):
    def _get_rule_str(self, node: Union[_Tree, _Token]):
        rule_str = super()._get_rule_str(node)
        return START_SYMBOL + ' ' + rule_str if self.add_start_end_toks else rule_str

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        Tree, Token = lark.Tree, lark.Token
        tree: Tree = example.get(self.source_key)
        if tree is not None:
            for node in self.traverse_tree(tree):
                rule_str = self._get_rule_str(node)
                yield from product([self.ns], rule_str.split())

        if self.add_start_end_toks:
            yield self.ns, END_SYMBOL

    def to_tensor(self, example) -> Mapping[str, Optional[torch.Tensor]]:
        tree: _Tree = example.get(self.source_key)
        if tree is None:
            return {self.renamed_key: None}

        tgt_raw = sum((self._get_rule_str(n).split() for n in self.traverse_tree(tree)), start=[])
        tgt = [self.vocab.get_token_index(s, self.ns) for s in tgt_raw]

        if self.add_start_end_toks:
            end_id = self.vocab.get_token_index(END_SYMBOL, self.ns)
            tgt += [end_id]

        if self.max_seq_len > 0:
            tgt = tgt[:self.max_seq_len]

        tgt = torch.tensor(tgt)
        return {self.renamed_key: tgt}


