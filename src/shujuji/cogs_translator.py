from trialbot.training import Registry
from trialbot.data.translator import FieldAwareTranslator
from trialbot.data.fields import SeqField


@Registry.translator('cogs-s2s')
class COGSTranslator(FieldAwareTranslator):
    def __init__(self):
        super().__init__(
            field_list=[
                SeqField('nl', 'source_tokens', add_start_end_toks=False,
                         max_seq_len=50, use_lower_case=False),
                SeqField('lf', 'target_tokens', use_lower_case=False)
            ],
        )
