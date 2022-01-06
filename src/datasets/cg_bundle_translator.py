from trialbot.training import Registry
from trialbot.data.translator import FieldAwareTranslator
from trialbot.data.fields import SeqField
from trialbot.data.translators import KnownFieldTranslator
from .cg_bundle_fields import TerminalRuleSeqField, ProcessedSentField
from .cfq_fields import TreeTraversalField, TutorBuilderField
from .cfq_fields import PolicyValidity


@Registry.translator('s2s')
class SQLSeq(FieldAwareTranslator):
    def __init__(self):
        super().__init__(field_list=[
            SeqField(source_key='sql', renamed_key="target_tokens",),
            SeqField(source_key='sent', renamed_key='source_tokens', add_start_end_toks=False,)
        ])


@Registry.translator('tranx')
class NoTermTranXTranslator(FieldAwareTranslator):
    def __init__(self):
        super().__init__(field_list=[
            ProcessedSentField(source_key='sent', renamed_key='source_tokens', add_start_end_toks=False,),
            TerminalRuleSeqField(no_preterminals=True,
                                 source_key='sql_tree', renamed_key="target_tokens", namespace="rule_seq", ),
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
