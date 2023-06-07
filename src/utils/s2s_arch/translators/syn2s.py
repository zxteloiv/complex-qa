# syntactic tree to sequence
from typing import Mapping, List, Optional, Generator, Tuple, Callable
from trialbot.data.field import Field, NullableTensor
from trialbot.data.fields import SeqField
from trialbot.data.translator import FieldAwareTranslator

from utils.preprocessing import nested_list_numbers_to_tensors
from utils.tree import PreOrderTraverse, Tree
from .plm2s import PREPROCESS_HOOKS


class BeNeParField(Field):
    def batch_tensor_by_key(self, tensors_by_keys: Mapping[str, List[NullableTensor]]
                            ) -> Mapping[str, 'torch.Tensor']:
        tokens_batch = tensors_by_keys[self.token_key]
        graph_batch = tensors_by_keys[self.graph_key]
        tokens = nested_list_numbers_to_tensors(tokens_batch, self.padding)
        graphs = nested_list_numbers_to_tensors(graph_batch, self.padding)
        return {self.token_key: tokens, self.graph_key: graphs}

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        sent = self.get_sent(example)
        sent = self.process_sent(sent)
        if sent is not None:
            tree = self.sent_to_tree(sent)
            for node in PreOrderTraverse()(tree):
                node: Tree
                yield self.ns, node.label

    def to_tensor(self, example):
        sent = self.get_sent(example)
        sent = self.process_sent(sent)
        if sent is None:
            return {self.token_key: None, self.graph_key: None}

        tree = self.sent_to_tree(sent)
        tokens = [self.vocab.get_token_index(node.label, self.ns) for node in PreOrderTraverse()(tree)]
        node_num = len(tokens)

        edges = []
        for node in PreOrderTraverse()(tree):
            node: Tree
            children_ids = set(c.node_id for c in node.children)
            edge = [1 if i in children_ids else 0 for i in range(node_num)]
            if node.parent is not None:
                edge[node.parent.node_id] = 1
            edges.append(edge)

        return {self.token_key: tokens, self.graph_key: edges}

    def sent_to_tree(self, sent: str) -> Tree:
        doc = self.nlp(sent)
        sents = list(doc.sents)
        root = sents[0]
        tree = self.span_to_tree(root)
        for sent in sents[1:]:
            tree.children.append(self.span_to_tree(sent))
        tree.assign_node_id(PreOrderTraverse())
        tree.build_parent_link()
        return tree

    @classmethod
    def span_to_tree(cls, node: 'spacy.tokens.Span') -> Tree:
        children = list(node._.children)
        labels = list(node._.labels)
        if len(children) == 0 or len(labels) == 0:
            return Tree(node.text, is_terminal=True)

        return Tree(labels[0], is_terminal=False, children=[
            cls.span_to_tree(c) for c in children
        ])

    def get_sent(self, example: dict):
        sent: Optional[str] = example.get(self.source_key)
        return sent

    def process_sent(self, sent: Optional[str]):
        if sent is None:
            return None

        if self.preprocess_hooks is not None:
            for hook in self.preprocess_hooks:
                sent = hook(sent)

        if self.use_lower_case:
            sent = sent.lower()

        return sent

    def __init__(self, source_key: str,
                 token_key: str,
                 graph_key: str,
                 namespace: str = None,
                 preprocess_hooks: Optional[PREPROCESS_HOOKS] = None,
                 padding_id: int = 0,
                 use_lower_case: bool = True,
                 spacy_model: str = 'en_core_web_md',
                 benepar_model: str = 'benepar_en3',
                 ):
        super().__init__()
        import spacy
        import benepar
        self.source_key = source_key
        self.ns = namespace or source_key
        self.token_key = token_key
        self.graph_key = graph_key
        self.preprocess_hooks = preprocess_hooks
        self.padding = padding_id
        self.use_lower_case = use_lower_case
        self.nlp = spacy.load(spacy_model)
        self.nlp.add_pipe('benepar', config={'model': benepar_model})


class Syn2SeqTranslator(FieldAwareTranslator):
    def __init__(self,
                 source_field: str,
                 target_field: str,
                 target_max_token: int = 0,
                 use_lower_case: bool = True,
                 source_preprocess_hooks: Optional[List[Callable[[str], str]]] = None,
                 spacy_model: str = 'en_core_web_md',
                 benepar_model: str = 'benepar_en3',
                 ):
        super().__init__([
            BeNeParField(source_field,
                         token_key='source_tokens',
                         graph_key='source_graph',
                         namespace='source_tokens',
                         preprocess_hooks=source_preprocess_hooks,
                         use_lower_case=use_lower_case,
                         spacy_model=spacy_model,
                         benepar_model=benepar_model,
                         ),

            SeqField(source_key=target_field,
                     renamed_key="target_tokens",
                     namespace="target_tokens",
                     max_seq_len=target_max_token,
                     use_lower_case=use_lower_case
                     ),
        ])
