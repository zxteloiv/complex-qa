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
                 padding: str = 0,
                 ):
        super().__init__()
        self.pretrained_model = AutoModel.from_pretrained(model_name)
        self.padding = padding

    def forward(self, model_input) -> Tuple[List[torch.Tensor], torch.Tensor]:
        # model input is actually a UserDict recognizable by the transformers
        out = self.pretrained_model(**model_input)
        mask = self.get_mask(model_input)
        return [out.last_hidden_state], mask

    def get_mask(self, model_input):
        return (model_input['input_ids'] != self.padding).long()
