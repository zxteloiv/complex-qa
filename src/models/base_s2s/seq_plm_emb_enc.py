from typing import Tuple, List

import torch.nn

from models.interfaces.encoder import EmbedAndEncode
from transformers import AutoModel


class SeqPLMEmbedEncoder(EmbedAndEncode):
    def is_bidirectional(self):
        return False

    def get_output_dim(self) -> int:
        return self.pretrained_model.config.hidden_size

    def __init__(self,
                 model_name: str = 'bert-base-uncased',
                 ):
        super().__init__()
        self.pretrained_model = AutoModel.from_pretrained(model_name)

    def forward(self, model_input) -> Tuple[List[torch.Tensor], torch.Tensor]:
        # model input is actually a UserDict recognizable by the transformers
        out = self.pretrained_model(**model_input)
        mask = (model_input['input_ids'] != 0).long()
        return [out.last_hidden_state], mask
