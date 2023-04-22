from typing import Optional, List, Callable
from .s2prod import TerminalRuleSeqField
from .plm2s import AutoPLMField, PREPROCESS_HOOKS
from trialbot.data.translator import FieldAwareTranslator


class PLM2ProdTranslator(FieldAwareTranslator):
    def __init__(self,
                 auto_plm_name: str,
                 source_field: str,
                 target_field: str,
                 target_max_token: int = 0,
                 use_lower_case: bool = True,
                 source_preprocess_hooks: Optional[PREPROCESS_HOOKS] = None,
                 no_preterminals: bool = True,
                 **auto_tokenizer_kwargs,
                 ):
        super().__init__([
            AutoPLMField(source_key=source_field,
                         auto_plm_name=auto_plm_name,
                         renamed_key='source_tokens',
                         preprocess_hooks=source_preprocess_hooks,
                         **auto_tokenizer_kwargs,
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
