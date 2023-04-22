# targeted at the model.base_seq2seq usage

from trialbot.data.translator import FieldAwareTranslator
from trialbot.data.fields import SeqField


class Seq2SeqTranslator(FieldAwareTranslator):
    def __init__(self,
                 source_field: str,
                 target_field: str,
                 source_max_token: int = 0,
                 target_max_token: int = 0,
                 use_lower_case: bool = True,
                 ):
        super().__init__(
            field_list=[
                SeqField(source_field,
                         renamed_key='source_tokens',
                         namespace='source_tokens',
                         add_start_end_toks=False,
                         max_seq_len=source_max_token,
                         use_lower_case=use_lower_case),
                SeqField(target_field,
                         renamed_key='target_tokens',
                         namespace='target_tokens',
                         max_seq_len=target_max_token,
                         use_lower_case=use_lower_case)
            ],
        )

