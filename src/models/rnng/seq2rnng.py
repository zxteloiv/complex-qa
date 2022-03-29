import torch
from torch import nn

from models.interfaces.encoder import EmbedAndEncode
from models.interfaces.loss_module import LossModule
from models.interfaces.metrics_module import MetricsModule
from models.interfaces.attention import Attention as IAttn, VectorContextComposer as AttnComposer
from models.rnng.rnng import RNNG
from utils.nn import aggregate_layered_state, assign_stacked_states


class Seq2RNNG(nn.Module):
    def __init__(self,
                 embed_encoder: EmbedAndEncode,
                 rnng: RNNG,
                 enc_dec_mapping: nn.Linear,
                 attn: IAttn,
                 attn_comp: AttnComposer,
                 init_strategy: str = 'forward_last_all',
                 ):
        super().__init__()
        self.embed_encoder = embed_encoder
        self.enc_dec_mapping = enc_dec_mapping
        self.rnng = rnng
        self.attn = attn
        self.attn_comp = attn_comp
        self.item_count = 0
        self.init_strategy = init_strategy

    def forward(self,
                source_tokens: torch.LongTensor,
                actions: torch.LongTensor = None,
                target_tokens: torch.LongTensor = None,
                ):
        layered_state, state_mask = self.embed_encoder(source_tokens)
        self._prepare_dec(layered_state, state_mask)

        preds, logits = self.rnng.forward(source_tokens.size()[0], actions, target_tokens)
        output = {'source': source_tokens,
                  'target': target_tokens,
                  'logits': logits,
                  'predictions': preds}

        if self.training:
            output['loss'] = self._compute_loss()
        self._compute_metrics(state_mask)
        return output

    def _prepare_dec(self, layered_state, state_mask):
        # force a consistent transformation
        layered_state = [self.enc_dec_mapping(x) for x in layered_state]
        src_agg = aggregate_layered_state(layered_state, state_mask, self.init_strategy)
        tgt_init = assign_stacked_states(
            src_agg, self.rnng.buffer_encoder.get_layer_num(), self.init_strategy
        )
        attn_and_comp_fn = self._get_attn_and_comp_fn(layered_state[-1], state_mask)
        self.rnng.set_conditions(tgt_init, attn_and_comp_fn)

    def _get_attn_and_comp_fn(self, state, mask):
        def attn_comp_fn(vec):
            if self.attn is None or self.attn_comp is None:
                return vec

            context = self.attn(vec, state, mask)
            return self.attn_comp.forward(context, vec)
        return attn_comp_fn

    def _compute_metrics(self, state_mask):
        self.item_count += (state_mask.sum(-1) > 0).sum().item()

    def _compute_loss(self):
        loss = self.rnng.get_loss()
        if isinstance(self.embed_encoder, LossModule):
            loss = loss + self.embed_encoder.get_loss()
        return loss

    def get_metrics(self, reset: bool = False):
        metrics = {"COUNT": self.item_count}
        if reset:
            self.item_count = 0

        if isinstance(self.rnng, MetricsModule):
            metrics.update(self.rnng.get_metrics(reset))

        if isinstance(self.embed_encoder, MetricsModule):
            metrics.update(self.embed_encoder.get_metrics(reset))

        return metrics

