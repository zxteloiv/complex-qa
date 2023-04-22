from typing import Mapping, List, Optional, Generator, Tuple, Callable
from .syn2s import BeNeParField
from .s2prod import TerminalRuleSeqField
from .plm2s import PREPROCESS_HOOKS
from trialbot.data.translator import FieldAwareTranslator


class Syn2ProdTranslator(FieldAwareTranslator):
    def __init__(self,
                 source_field: str,
                 target_field: str,
                 target_max_token: int = 0,
                 use_lower_case: bool = True,
                 source_preprocess_hooks: Optional[PREPROCESS_HOOKS] = None,
                 no_preterminals: bool = True,
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

            TerminalRuleSeqField(keywords=None,
                                 no_preterminals=no_preterminals,
                                 source_key=target_field,
                                 renamed_key='target_tokens',
                                 namespace='target_tokens',
                                 max_seq_len=target_max_token,
                                 add_start_end_toks=True,
                                 use_lower_case=use_lower_case,
                                 ),
        ])
