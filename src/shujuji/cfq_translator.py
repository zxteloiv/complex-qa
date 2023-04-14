from trialbot.training import Registry
from utils.sparql_tokenizer import split_sparql
from trialbot.data.translator import FieldAwareTranslator
from trialbot.data.fields import SeqField
from .cfq_fields import TreeTraversalField, GrammarPatternSeqField, GrammarModEntSeqField, TutorBuilderField

@Registry.translator('cfq_seq_mod_ent_qa')
class CFQSeq(FieldAwareTranslator):
    def __init__(self):
        super().__init__(field_list=[
            SeqField(source_key='sparqlPatternModEntities', renamed_key="target_tokens", split_fn=split_sparql,),
            SeqField(source_key='questionPatternModEntities', renamed_key='source_tokens', add_start_end_toks=False,)
        ])

# namespaces definition and the corresponding fidelity
TREE_NS: str = 'symbol'

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

@Registry.translator('cfq_flat_pda')
class CFQFlatDerivations(FieldAwareTranslator):
    def __init__(self, max_derivation_symbols: int = 11):
        super().__init__(field_list=[
            SeqField(source_key='questionPatternModEntities', renamed_key='source_tokens', add_start_end_toks=False,),
            TreeTraversalField(tree_key='sparqlPatternModEntities_tree', namespace=TREE_NS, )
        ])

class CFQTutorBuilder(FieldAwareTranslator):
    def __init__(self):
        super().__init__(field_list=[
            TutorBuilderField(tree_key='sparqlPatternModEntities_tree', ns=TREE_NS)
        ])