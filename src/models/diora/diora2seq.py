import logging
from typing import Dict
import torch
from utils.nn import assign_stacked_states
from ..base_s2s.base_seq2seq import BaseSeq2Seq
from .diora_encoder import DioraEncoder, EncoderRNNStack
from utils.nn import prepare_input_mask, seq_cross_ent
from .hard_diora import DioraTopk
from .diora import Diora
from functools import reduce


class Diora2Seq(BaseSeq2Seq):
    def __init__(self, use_diora_loss: bool = False, one_by_one: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self._encoder, DioraEncoder):
            raise TypeError("Diora2Seq instances must be assembled with"
                            "DioraEncoder instead of other StackedEncoder."
                            "Because it is forbidden to stack Diora with other type of encoders.")
        if self._strategy.startswith("forward_last"):
            logging.getLogger(self.__class__.__name__).info(
                f'the strategy {self._strategy} will choose the last position of the diora encoder '
                f'output, which is actually the root of the source input, to initialize the decoder '
                f'states.'
            )
        else:
            logging.getLogger(self.__class__.__name__).info(
                f'the strategy {self._strategy} may use all the positions along with all the constituents '
                f'to aggregate the encoder representation, and then to initialize the decoder.'
            )
        # use the settings similar to the original implementation, fixed embedding and train the projections;
        # there's already a leaf transformation within diora, so we only project the embeddings for the loss
        # src_vocab_size = self.vocab.get_vocab_size(self._source_namespace)
        _, emb_sz = self._src_embedding.weight.size()
        self.reconstruct = torch.nn.Linear(self._encoder.diora.size, emb_sz)
        # as shown in the de-attentions setup, the embedding is not kept fixed
        # self._src_embedding.weight.requires_grad = False
        self.use_diora_loss = use_diora_loss
        self.one_by_one = one_by_one

    def _get_reconstruct_loss(self, source, mask):
        self._encoder: DioraEncoder
        diora = self._encoder.diora
        batch_sz, length = mask.size()
        # cells: (batch, length, hidden)
        cells = diora.chart['outside_h'][:, :length]

        # proj_cells: (batch, length, emb_sz)
        proj_cells = self.reconstruct(cells)
        # emb_weight: (num_toks, emb_sz)
        emb_weight = self._src_embedding.weight
        # logits: (batch, length, num_toks)
        logits = torch.einsum('ble,ne->bln', proj_cells, emb_weight)

        loss = seq_cross_ent(logits, source, mask, average="token")
        return loss

    def _get_tree_loss(self, mask: torch.Tensor, margin=1):
        self._encoder: DioraEncoder
        # root_idx: (batch,)
        root_idx = self._encoder.get_root_cell_idx(mask)
        batch_sz = mask.size()[0]
        diora = self._encoder.diora
        batch_idx = torch.arange(batch_sz, device=mask.device)
        gold_score = diora.chart['inside_s'][batch_idx, root_idx].contiguous().view(batch_sz, 1)
        pred_score = diora.charts[1]['inside_s'][batch_idx, root_idx].contiguous().view(batch_sz, 1)

        # just follows the original definition
        tr_loss = torch.clamp(pred_score + margin - gold_score, min=0)
        # returned tr_loss: (,) <- (batch, 1)
        return tr_loss.mean()

    def forward(self, source_tokens: torch.LongTensor,
                target_tokens: torch.LongTensor = None
                ) -> Dict[str, torch.Tensor]:
        if self.one_by_one:
            return self._forward_one_by_one(source_tokens, target_tokens)
        else:
            return self._forward_batch(source_tokens, target_tokens)

    def _forward_one_by_one(self, source_tokens: torch.LongTensor, target_tokens: torch.LongTensor):
        # simulated for 1-size batch iteration
        outputs = []
        for batch_id, src_toks in enumerate(source_tokens):
            # batch_id: int
            # src_toks: (max_src_len,)

            valid_src_len = (src_toks > 0).sum()
            src_toks = src_toks[:valid_src_len].unsqueeze(0)

            # tgt_toks: (max_tgt_len,)
            tgt_toks = None
            if target_tokens is not None:
                tgt_toks = target_tokens[batch_id]
                valid_tgt_len = (tgt_toks > 0).sum()
                tgt_toks = tgt_toks[:valid_tgt_len].unsqueeze(0)

            output = self._forward_batch(src_toks, tgt_toks)
            outputs.append(output)

        final_output = {"batch_output": outputs}
        if len(outputs) > 0 and 'loss' in outputs[0]:
            batch_sz = len(outputs)
            final_output['loss'] = reduce(lambda x, y: x + y, [o['loss'] for o in outputs], 0) / batch_sz

        return final_output

    def _forward_batch(self, source_tokens, target_tokens):
        output = super().forward(source_tokens, target_tokens)
        if self.use_diora_loss and self.training and 'loss' in output:
            src, src_mask = prepare_input_mask(source_tokens, self._padding_index)
            reconstruct_loss = self._get_reconstruct_loss(src, src_mask)
            output['loss'] = output['loss'] + reconstruct_loss

            self._encoder: DioraEncoder
            if isinstance(self._encoder.diora, DioraTopk):
                tr_loss = self._get_tree_loss(src_mask, margin=1)
                output['loss'] = output['loss'] + tr_loss

        return output

    def _forward_enc(self, source_tokens):
        # source: (batch, source_length), containing the input word IDs
        # target: (batch, target_length), containing the output IDs
        source, source_mask = prepare_input_mask(source_tokens, self._padding_index)
        chart_output, layered_states = self._encode(source, source_mask)
        # afterwards, source tokens are not used anymore and only the selected diora charts are required
        self._encoder: DioraEncoder
        chart_mask = self._encoder.output_mask
        return source, source_mask, chart_output, layered_states, chart_mask

    @staticmethod
    def get_encoder(p) -> 'EncoderRNNStack':
        diora_type = p.encoder
        assert diora_type in ('diora', 's-diora'), f'unsupported diora type {diora_type}'

        diora_cls = {
            'diora': Diora,
            's-diora': DioraTopk,
        }.get(diora_type)
        hid_sz = getattr(p, 'enc_out_dim', p.hidden_sz)
        diora_kwargs = getattr(p, 'diora_kwargs', dict(size=hid_sz, input_size=p.emb_sz))
        diora = diora_cls.from_kwargs_dict(diora_kwargs)
        enc = DioraEncoder(diora, getattr(p, 'diora_concat_outside', False))
        return enc

    @classmethod
    def from_param_and_vocab(cls, p, vocab):
        model = super().from_param_and_vocab(p, vocab)
        model.use_diora_loss = getattr(p, 'diora_loss_enabled', False)
        return model
