from trialbot.training import Registry
from trialbot.data.translator import FieldAwareTranslator
from .cg_bundle_fields import ProcessedSentField, RNNGField
from .cfq_fields import TreeTraversalField, TutorBuilderField
from .cfq_fields import PolicyValidity
from utils.s2s_arch.translators import PLM2SeqTranslator, Seq2SeqTranslator, Seq2ProdTranslator, PLM2ProdTranslator, \
    Syn2SeqTranslator, Syn2ProdTranslator


@Registry.translator('s2s')
class SQLSeq(Seq2SeqTranslator):
    def __init__(self):
        super().__init__('sql', 'sent', 20, 200,)


@Registry.translator('plm2s')
class PLMSeq(PLM2SeqTranslator):
    def __init__(self, model_name: str = 'bert-base-uncased'):
        super().__init__(model_name, 'sent', 'sql', 200,
                         source_preprocess_hooks=[ProcessedSentField.process_sentence])


@Registry.translator('plm2tranx')
class PLMTranX(PLM2ProdTranslator):
    def __init__(self, model_name: str = 'bert-base-uncased'):
        super().__init__(
            model_name, 'sent', 'sql_tree', 280,
            source_preprocess_hooks=[ProcessedSentField.process_sentence],
            no_preterminals=True,
        )


@Registry.translator('syn2s')
class SynSeq(Syn2SeqTranslator):
    def __init__(self):
        super().__init__('sent', 'sql', 200,
                         source_preprocess_hooks=[ProcessedSentField.process_sentence]
                         )


@Registry.translator('syn2tranx')
class SynTranX(Syn2ProdTranslator):
    def __init__(self):
        super().__init__(
            'sent', 'sql_tree', 280,
            source_preprocess_hooks=[ProcessedSentField.process_sentence],
            no_preterminals=True,
        )


@Registry.translator('tranx')
class TranXTranslator(Seq2ProdTranslator):
    def __init__(self):
        super().__init__('sent', 'sql_tree', 20, 280, no_preterminals=True)


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
