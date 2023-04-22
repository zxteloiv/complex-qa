from typing import Union, Optional, Tuple, Dict, Any, Literal, List
import torch
from models.base_s2s.base_seq2seq import BaseSeq2Seq
from models.interfaces.encoder import EmbedAndGraphEncode


class SynGraph2Seq(BaseSeq2Seq):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self._embed_encoder, EmbedAndGraphEncode)

    def forward(self,
                source_tokens: torch.LongTensor,
                source_graph: torch.Tensor = None,
                target_tokens: torch.LongTensor = None,
                **kwargs,
                ) -> Dict[str, torch.Tensor]:
        self._reset_variational_dropouts()
        self._embed_encoder: EmbedAndGraphEncode
        self._embed_encoder.set_graph(source_graph)
        layer_states, state_mask = self._embed_encoder(source_tokens)
        hx, enc_attn_fn, start = self._prepare_dec(layer_states, state_mask.long())
        preds, logits = self._forward_dec(target_tokens, start, enc_attn_fn, hx)

        output = {}
        if self.training:
            output['loss'] = self._compute_loss(logits, target_tokens, state_mask)

        if target_tokens is not None:
            total_err = self._compute_metrics(source_tokens, target_tokens, preds, logits)
            output.update(errno=total_err.tolist())

        output.update(predictions=preds, logits=logits, target=target_tokens)
        return output

