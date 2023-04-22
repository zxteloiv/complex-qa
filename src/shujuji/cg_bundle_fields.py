from typing import List, Mapping, Generator, Tuple
import torch
from trialbot.data.fields import SeqField
from trialbot.data.field import Field, NullableTensor
from trialbot.data import START_SYMBOL, END_SYMBOL
import nltk
import re
from utils.tree import Tree, PreorderTraverse
from utils.preprocessing import nested_list_numbers_to_tensors
from models.rnng import rnng_utils


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


class RNNGField(Field):
    def batch_tensor_by_key(self, tensors_by_keys: Mapping[str, List[NullableTensor]]) -> Mapping[str, torch.Tensor]:
        action_list, target_list = list(map(tensors_by_keys.get, (self.action_key, self.target_key)))
        if any(x is None or len(x) == 0 for x in (action_list, target_list)):
            raise ValueError(f'input is empty or contains null keys: {self.action_key}, {self.target_key}')

        return {
            self.action_key: nested_list_numbers_to_tensors(action_list),
            self.target_key: nested_list_numbers_to_tensors(target_list),
        }

    def _check_root(self, node: Tree):
        if self._default_root is not None and node.label != self._default_root:
            raise ValueError(f'root ({node.label}) inconsistent with the default ({self._default_root})')

        if node.is_terminal:
            raise ValueError(f'root ({node.label}) is not a non-terminal')

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        tree: Tree = example.get(self.source_key)
        if tree is not None:
            # set root and emit for the root namespace
            self._check_root(tree)
            self._default_root = tree.label
            yield self.ns_root, rnng_utils.get_token_str(tree)
            # manually emit reduce action for RNNG, thus we can get rid of the hooks
            yield self.ns_rnng, rnng_utils.get_token_str(rnng_utils.get_reduce_token())
            yield self.ns_term, START_SYMBOL    # start token to initialize the output buffer
            for node in PreorderTraverse()(tree):
                node: Tree
                # since nodes are either terminals or nonterminals,
                # they commit to the ns_nt or ns_term namespace.
                # the reduce action is special and not arisen direct by tokens.
                ns = self.ns_nt if rnng_utils.is_nt_action(node) else self.ns_term
                yield ns, rnng_utils.get_token_str(node)

    def to_tensor(self, example) -> Mapping[str, NullableTensor]:
        tree: Tree = example.get(self.source_key)
        if tree is None:
            return {self.action_key: None, self.target_key: None}

        def _add_reduce_action(n: Tree, *args):
            return [rnng_utils.get_reduce_token()] if not n.is_terminal else []

        traverse = PreorderTraverse(hooks={'post_children': _add_reduce_action})
        actions, target = [], [self.token_to_rnng_id(START_SYMBOL)]
        for node in traverse(tree):
            actions.append(self.token_to_rnng_id(node))
            if rnng_utils.is_gen_action(node):
                target.append(self.token_to_rnng_id(node))
        return {self.action_key: actions, self.target_key: target}

    def token_to_rnng_id(self, tok: Tree):
        return rnng_utils.token_to_id(tok, self.vocab, (self.ns_rnng, self.ns_nt, self.ns_term))

    def __init__(self,
                 source_key: str,
                 action_key: str = 'actions',
                 target_key: str = 'target_tokens',
                 ns_terminals='term',
                 ns_non_terminals='cat',
                 ns_rnng='rnng',
                 ns_root='grammar_entry',
                 ):
        super().__init__()
        self.source_key = source_key
        self.ns_nt = ns_non_terminals
        self.ns_term = ns_terminals
        self.ns_rnng = ns_rnng
        self.ns_root = ns_root
        self.action_key = action_key
        self.target_key = target_key
        self._default_root = None
