from trialbot.training import Registry
from trialbot.data.translator import FieldAwareTranslator
from trialbot.data.fields import SeqField
from trialbot.data.translators import KnownFieldTranslator
from .cg_bundle_fields import TerminalRuleSeqField, ProcessedSentField, RNNGField, BeNeParField
from .cfq_fields import TreeTraversalField, TutorBuilderField
from .cfq_fields import PolicyValidity


@Registry.translator('s2s')
class SQLSeq(FieldAwareTranslator):
    def __init__(self):
        super().__init__(field_list=[
            SeqField(source_key='sql', renamed_key="target_tokens",),
            SeqField(source_key='sent', renamed_key='source_tokens', add_start_end_toks=False,)
        ])


@Registry.translator('syn2s')
class SynSeq(FieldAwareTranslator):
    def __init__(self):
        super().__init__(field_list=[
            BeNeParField(source_key='sent', token_key='source_tokens', graph_key='source_graph',
                         preprocess_hooks=[ProcessedSentField.process_sentence],),
            SeqField(source_key='sql', renamed_key="target_tokens", ),
        ])


@Registry.translator('syn2tranx')
class SynTranX(FieldAwareTranslator):
    def __init__(self):
        super().__init__(field_list=[
            BeNeParField(source_key='sent', token_key='source_tokens', graph_key='source_graph',
                         preprocess_hooks=[ProcessedSentField.process_sentence],),
            TerminalRuleSeqField(no_preterminals=True,
                                 source_key='sql_tree', renamed_key="target_tokens", namespace="rule_seq", ),
        ])


@Registry.translator('tranx')
class NoTermTranXTranslator(FieldAwareTranslator):
    def __init__(self):
        super().__init__(field_list=[
            ProcessedSentField(source_key='sent', renamed_key='source_tokens', add_start_end_toks=False,),
            TerminalRuleSeqField(no_preterminals=True,
                                 source_key='sql_tree', renamed_key="target_tokens", namespace="rule_seq", ),
        ])


@Registry.translator('rnng')
class RNNGTranslator(FieldAwareTranslator):
    def __init__(self, src_ns='sent',
                 rnng_namespaces=('rnng', 'nonterminal', 'terminal'),
                 grammar_entry_ns: str = 'grammar_entry',):
        super().__init__(field_list=[
            ProcessedSentField(source_key=src_ns, renamed_key='source_tokens', add_start_end_toks=False,),
            RNNGField(source_key='runtime_tree',
                      action_key='actions',         # follow the param name of the RNNG model
                      target_key='target_tokens',
                      ns_rnng=rnng_namespaces[0],
                      ns_non_terminals=rnng_namespaces[1],
                      ns_terminals=rnng_namespaces[2],
                      ns_root=grammar_entry_ns,
                      ),
        ])


TREE_NS = 'symbol'


@Registry.translator('cg_sql_pda')
class SQLDerivations(FieldAwareTranslator):
    def __init__(self, use_reversible_actions: bool = True):
        super().__init__(field_list=[
            ProcessedSentField(source_key='sent', renamed_key='source_tokens', add_start_end_toks=False,),
            TreeTraversalField(tree_key='runtime_tree', namespace=TREE_NS,),
            PolicyValidity(tree_key='runtime_tree', use_reversible_actions=use_reversible_actions),
        ])

    def batch_tensor(self, tensors):
        tensors = list(filter(lambda x: x.get('tree_nodes') is not None, tensors))
        batch_dict = self.list_of_dict_to_dict_of_list(tensors)
        output = {}
        for field in self.fields:
            try:
                output.update(field.batch_tensor_by_key(batch_dict))
            except KeyError:
                return None

        return output


class SQLTutorBuilder(FieldAwareTranslator):
    def __init__(self):
        super().__init__(field_list=[
            TutorBuilderField(tree_key='runtime_tree', ns=TREE_NS)
        ])
