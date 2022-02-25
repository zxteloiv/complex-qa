from typing import Dict
import torch
from utils.nn import assign_stacked_states
from ..base_s2s.base_seq2seq import BaseSeq2Seq
from .diora_encoder import DioraEncoder, EncoderRNNStack
from utils.nn import prepare_input_mask, seq_cross_ent
from .hard_diora import DioraMLPWithTopk
from .diora import DioraMLP


class Diora2Seq(BaseSeq2Seq):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self._encoder, DioraEncoder):
            raise TypeError("Diora2Seq instances must be assembled with"
                            "DioraEncoder instead of other StackedEncoder."
                            "Because it is forbidden to stack Diora with other type of encoders.")

        # use the settings similar to the original implementation, fixed embedding and train the projections;
        # there's already a leaf transformation within diora, so we only project the embeddings for the loss
        # src_vocab_size = self.vocab.get_vocab_size(self._source_namespace)
        _, emb_sz = self._src_embedding.weight.size()
        self.reconstruct = torch.nn.Linear(self._encoder.get_output_dim(), emb_sz)
        # self._src_embedding.weight.requires_grad = False

    def _init_decoder(self, layer_states, source_mask):
        usage: str = self._enc_dec_trans_usage
        # for Diora, the encoder must contain barely 1 layer
        assert len(layer_states) == 1 and layer_states[0].size()[1] == 1
        agg_src = [s[:, 0] for s in layer_states]

        if usage == 'dec_init':
            agg_src = list(map(self._enc_dec_trans, agg_src))

        # "avg_all" is only meaningful at the "_all" part, but the interface requires both
        # _all is a fixed setting for diora but hardly matters because the target decoder
        # seldom uses more than 1 layer
        dec_states = assign_stacked_states(agg_src, self._decoder.get_layer_num(), 'avg_all')
        hx, _ = self._decoder.init_hidden_states(dec_states)
        return hx

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

        # duplicate_mask = self._build_duplicate_token_mask(source)
        duplicate_mask = 1
        loss = seq_cross_ent(logits, source, duplicate_mask * mask, average="token")
        return loss

    def _build_duplicate_token_mask(self, source: torch.Tensor):
        # build duplicate mask
        batch_sz, length = source.size()
        duplicate_mask = torch.zeros_like(source)
        mini_batch_toks = set()
        for b in range(batch_sz):
            for l in range(length):
                tok = source[b, l].item()
                if tok not in mini_batch_toks:
                    duplicate_mask[b, l] = 1
                    mini_batch_toks.add(tok)
        return duplicate_mask

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
        output = super().forward(source_tokens, target_tokens)
        if not self.training or 'loss' not in output:
            return output

        source, source_mask = prepare_input_mask(source_tokens, self._padding_index)
        reconstruct_loss = self._get_reconstruct_loss(source, source_mask)
        output['loss'] = output['loss'] + reconstruct_loss

        self._encoder: DioraEncoder
        if isinstance(self._encoder.diora, DioraMLPWithTopk):
            tr_loss = self._get_tree_loss(source_mask, margin=1)
            output['loss'] = output['loss'] + tr_loss

        return output

    @staticmethod
    def get_encoder(p) -> 'EncoderRNNStack':
        diora_type = p.encoder
        assert diora_type in ('diora', 's-diora'), f'unsupported diora type {diora_type}'

        diora_cls = {
            'diora': DioraMLP,
            's-diora': DioraMLPWithTopk,
        }.get(diora_type)
        num_beam = getattr(p, 'diora_topk', 2)
        hid_sz = getattr(p, 'enc_out_dim', p.hidden_sz)
        diora = diora_cls.from_kwargs_dict(dict(
            input_size=p.emb_sz, size=hid_sz, n_layers=2, K=num_beam,
        ))
        enc = DioraEncoder(diora)
        return enc
