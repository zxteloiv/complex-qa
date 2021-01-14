from trialbot.training import Registry
from trialbot.data import Translator, START_SYMBOL, END_SYMBOL, PADDING_TOKEN
from utils.sparql_tokenizer import split_sparql
from .field import FieldAwareTranslator
from .seq_field import SeqField
from .cfq_fields import MidOrderTraversalField, GrammarPatternSeqField, GrammarModEntSeqField, TutorBuilderField

@Registry.translator('cfq_seq_mod_ent_qa')
class CFQSeq(FieldAwareTranslator):
    def __init__(self):
        super().__init__(field_list=[
            SeqField(source_key='sparqlPatternModEntities', renamed_key="target_tokens", split_fn=split_sparql,),
            SeqField(source_key='questionPatternModEntities', renamed_key='source_tokens', add_start_end_toks=False,)
        ])

# namespaces definition and the corresponding fidelity
PARSE_TREE_NS = (NS_NT, NS_T, NS_ET) = ('nonterminal', 'terminal_category', 'terminal')
NS_FI = (NS_NT_FI, NS_T_FI, NS_ET_FI) = (0, 1, 2)
UNIFIED_TREE_NS = ('symbol', 'exact_token')

@Registry.translator('cfq_tranx_pattern_qa')
class TranXStyleTranslator(FieldAwareTranslator):
    def __init__(self):
        super().__init__(field_list=[
            SeqField(source_key='questionPatternModEntities', renamed_key='source_tokens', add_start_end_toks=False,),
            GrammarPatternSeqField(source_key='sparqlPatternModEntities_tree', renamed_key="target_tokens",
                                   namespace="modent_rule_seq", split_fn=split_sparql, ),
        ])

@Registry.translator('cfq_tranx_mod_ent_qa')
class TranXModEntTranslator(FieldAwareTranslator):
    def __init__(self):
        super().__init__(field_list=[
            SeqField(source_key='questionPatternModEntities', renamed_key='source_tokens', add_start_end_toks=False,),
            GrammarModEntSeqField(source_key='sparqlPatternModEntities_tree', renamed_key="target_tokens",
                                  namespace="modent_rule_seq", split_fn=split_sparql, ),
        ])

@Registry.translator('cfq_flat_pda_mod_ent_qa')
class CFQFlatDerivations(FieldAwareTranslator):
    def __init__(self, max_derivation_symbols: int = 11):
        super().__init__(field_list=[
            SeqField(source_key='questionPatternModEntities', renamed_key='source_tokens', add_start_end_toks=False,),
            MidOrderTraversalField(tree_key='sparqlPatternModEntities_tree',
                                   namespaces=UNIFIED_TREE_NS,
                                   max_derivation_symbols=max_derivation_symbols,
                                   )
        ])

class CFQTutorBuilder(FieldAwareTranslator):
    def __init__(self):
        super().__init__(field_list=[
            TutorBuilderField(tree_key='sparqlPatternModEntities_tree',
                              ns=UNIFIED_TREE_NS)
        ])