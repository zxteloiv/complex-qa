from typing import List, Mapping, Generator, Tuple, Optional, Any, Literal, Iterable, Union, DefaultDict
from trialbot.training import Registry
from .field import FieldAwareTranslator
from .seq_field import SeqField
from .cg_bundle_fields import TerminalRuleSeqField

@Registry.translator('sql_s2s')
class SQLSeq(FieldAwareTranslator):
    def __init__(self):
        super().__init__(field_list=[
            SeqField(source_key='sql', renamed_key="target_tokens",),
            SeqField(source_key='sent', renamed_key='source_tokens', add_start_end_toks=False,)
        ])

@Registry.translator('cg_sql_tranx')
class CGSQLTranXTranslator(FieldAwareTranslator):
    def __init__(self):
        super().__init__(field_list=[
            SeqField(source_key='sent', renamed_key='source_tokens', add_start_end_toks=False,),
            TerminalRuleSeqField(source_key='sql_tree', renamed_key="target_tokens", namespace="rule_seq",),
        ])

    def batch_tensor(self, tensors):
        tensors = list(filter(lambda x: x.get('target_tokens') is not None, tensors))
        batch_dict = self.list_of_dict_to_dict_of_list(tensors)
        output = {}
        for field in self.fields:
            try:
                output.update(field.batch_tensor_by_key(batch_dict))
            except:
                return None

        return output


# # namespaces definition and the corresponding fidelity
# PARSE_TREE_NS = (NS_NT, NS_T, NS_ET) = ('nonterminal', 'terminal_category', 'terminal')
# NS_FI = (NS_NT_FI, NS_T_FI, NS_ET_FI) = (0, 1, 2)
# UNIFIED_TREE_NS = ('symbol', 'exact_token')
#
# @Registry.translator('cg_sql_pda')
# class FlatDerivations(FieldAwareTranslator):
#     def __init__(self, max_derivation_symbols: int = 11):
#         super().__init__(field_list=[
#             SeqField(source_key='sent', renamed_key='source_tokens', add_start_end_toks=False,),
#             MidOrderTraversalField(tree_key='sparqlPatternModEntities_tree',
#                                    namespaces=UNIFIED_TREE_NS,
#                                    max_derivation_symbols=max_derivation_symbols,
#                                    )
#         ])
