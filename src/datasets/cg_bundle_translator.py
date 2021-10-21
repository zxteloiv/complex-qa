from trialbot.training import Registry
from trialbot.data.translator import FieldAwareTranslator
from trialbot.data.fields import SeqField
from trialbot.data.translators import KnownFieldTranslator
from .cg_bundle_fields import TerminalRuleSeqField, ProcessedSentField, RuleSymbolSeqField, TreeField, DerivationField
from .cfq_fields import MidOrderTraversalField, TutorBuilderField

@Registry.translator('sql_s2s')
class SQLSeq(FieldAwareTranslator):
    def __init__(self):
        super().__init__(field_list=[
            SeqField(source_key='sql', renamed_key="target_tokens",),
            SeqField(source_key='sent', renamed_key='source_tokens', add_start_end_toks=False,)
        ])

@Registry.translator('tranx')
class TranXTranslator(FieldAwareTranslator):
    def __init__(self):
        super().__init__(field_list=[
            ProcessedSentField(source_key='sent', renamed_key='source_tokens', add_start_end_toks=False, ),
            TerminalRuleSeqField(no_terminal_rule=False,
                                 source_key='sql_tree', renamed_key="target_tokens", namespace="rule_seq", ),
        ])

@Registry.translator('tranx_no_terminal')
class NoTermTranXTranslator(FieldAwareTranslator):
    def __init__(self):
        super().__init__(field_list=[
            ProcessedSentField(source_key='sent', renamed_key='source_tokens', add_start_end_toks=False,),
            TerminalRuleSeqField(no_terminal_rule=True,
                                 source_key='sql_tree', renamed_key="target_tokens", namespace="rule_seq",),
        ])

@Registry.translator('gd')
class GrammarDerivationTranslator(KnownFieldTranslator):
    def __init__(self, max_node_position=12):
        super().__init__(
            field_list=[
                ProcessedSentField(source_key='sent', renamed_key='source_tokens', add_start_end_toks=False,),
                DerivationField(source_key='runtime_tree', renamed_key="target_tokens", namespace="formal_token"),
                TreeField(tree_key="runtime_tree", ns="formal_token", max_node_position=max_node_position)
            ],
            vocab_fields=[
                ProcessedSentField(source_key='sent', renamed_key='source_tokens', add_start_end_toks=False,),
                DerivationField(source_key='sql_tree', renamed_key="target_tokens", namespace="formal_token"),
            ]
        )

# namespaces definition and the corresponding fidelity
PARSE_TREE_NS = (NS_NT, NS_T, NS_ET) = ('nonterminal', 'terminal_category', 'terminal')
NS_FI = (NS_NT_FI, NS_T_FI, NS_ET_FI) = (0, 1, 2)
UNIFIED_TREE_NS = ('symbol', 'exact_token')

@Registry.translator('cg_sql_pda')
class SQLDerivations(FieldAwareTranslator):
    def __init__(self, max_derivation_symbols: int = 11):
        super().__init__(field_list=[
            ProcessedSentField(source_key='sent', renamed_key='source_tokens', add_start_end_toks=False,),
            MidOrderTraversalField(tree_key='sql_tree',
                                   namespaces=UNIFIED_TREE_NS,
                                   max_derivation_symbols=max_derivation_symbols,
                                   grammar_token_generation=True,
                                   )
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
            TutorBuilderField(tree_key='sql_tree',
                              ns=UNIFIED_TREE_NS)
        ])
