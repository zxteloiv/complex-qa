import logging
from typing import Union, Optional, Tuple, Dict, Any, Literal, List
import numpy as np
import torch
from torch import nn
from trialbot.data.ns_vocabulary import NSVocabulary
from ..interfaces.attention import Attention as IAttn, VectorContextComposer as AttnComposer
from ..interfaces.loss_module import LossModule
from ..modules.variational_dropout import VariationalDropout
from ..interfaces.unified_rnn import RNNStack, UnifiedRNN
from ..interfaces.encoder import EmbedAndEncode
from utils.nn import filter_cat, prepare_input_mask, seq_cross_ent
from utils.nn import aggregate_layered_state, assign_stacked_states
from utils.seq_collector import SeqCollector
from utils.text_tool import make_human_readable_text
from allennlp.training.metrics import BLEU, Perplexity, Average


class BaseSeq2Seq(torch.nn.Module):
    def __init__(self,
                 # modules
                 vocab: NSVocabulary,
                 embed_encoder: EmbedAndEncode,
                 decoder: RNNStack,
                 word_projection: torch.nn.Module,
                 target_embedding: torch.nn.Embedding,
                 enc_attention: IAttn = None,
                 dec_hist_attn: IAttn = None,
                 dec_inp_attn_comp: AttnComposer = None,
                 proj_inp_attn_comp: AttnComposer = None,
                 enc_dec_transformer: torch.nn.Module = None,

                 # model configuration
                 source_namespace: str = "source_tokens",
                 target_namespace: str = "target_tokens",
                 start_symbol: str = '<GO>',
                 eos_symbol: str = '<EOS>',
                 padding_index: int = 0,
                 max_decoding_step: int = 50,
                 decoder_init_strategy: str = "forward_last_all",
                 enc_dec_transform_usage: str = 'consistent',

                 # training_configuration
                 scheduled_sampling_ratio: float = 0.,
                 dec_dropout: float = .1,
                 training_average: Literal["token", "batch", "none"] = "batch",
                 attn_supervision: str = "none",
                 ):
        super().__init__()
        self.vocab = vocab
        self._enc_attn = enc_attention
        self._dec_hist_attn = dec_hist_attn

        self._embed_encoder = embed_encoder

        self._dec_inp_attn_comp = dec_inp_attn_comp
        self._proj_inp_attn_comp = proj_inp_attn_comp

        self._scheduled_sampling_ratio = scheduled_sampling_ratio
        self._decoder = decoder
        self._tgt_embedding = target_embedding
        self._enc_dec_trans = enc_dec_transformer
        self._enc_dec_trans_usage = enc_dec_transform_usage

        self._start_id = vocab.get_token_index(start_symbol, target_namespace)
        self._eos_id = vocab.get_token_index(eos_symbol, target_namespace)
        self._max_decoding_step = max_decoding_step

        self._source_namespace = source_namespace
        self._target_namespace = target_namespace

        self._output_projection = word_projection

        self._tgt_emb_dropout = VariationalDropout(dec_dropout, on_the_fly=False)
        self._proj_inp_dropout = VariationalDropout(dec_dropout, on_the_fly=False)
        self.mem: Optional[SeqCollector] = None

        self._padding_index = padding_index
        self._strategy = decoder_init_strategy
        self._training_avg = training_average
        self._attn_sup = attn_supervision

        self.bleu = BLEU(exclude_indices={padding_index, self._start_id, self._eos_id})
        self.ppl = Perplexity()
        self.err_rate = Average()
        self.src_len = Average()
        self.tgt_len = Average()
        self.item_count = 0
        self.sparsity = Average()

    def get_metric(self, reset=False):
        metric = {"PPL": round(self.ppl.get_metric(reset), 6),
                  "ERR": round(self.err_rate.get_metric(reset), 6),
                  "COUNT": self.item_count,
                  "SLEN": round(self.src_len.get_metric(reset), 2),
                  "TLEN": round(self.tgt_len.get_metric(reset), 2),
                  "GINI": round(self.sparsity.get_metric(reset), 6),
                  }
        metric.update(self.bleu.get_metric(reset))
        if reset:
            self.item_count = 0

        return metric

    # for possible misspelling error
    get_metrics = get_metric

    def revert_tensor_to_string(self, output_dict: dict) -> dict:
        """Convert the predicted word ids into discrete tokens"""
        # predictions: (batch, max_length)
        output_dict['predicted_tokens'] = make_human_readable_text(
            output_dict['predictions'], self.vocab, self._target_namespace,
            stop_ids=[self._eos_id, self._padding_index],
        )
        output_dict['source_tokens'] = make_human_readable_text(
            output_dict['source'], self.vocab, self._source_namespace,
            stop_ids=[self._padding_index],
        )
        output_dict['target_tokens'] = make_human_readable_text(
            output_dict['target'][:, 1:], self.vocab, self._target_namespace,
            stop_ids=[self._eos_id, self._padding_index]
        )
        return output_dict

    def forward(self,
                source_tokens: Union[torch.LongTensor, dict],
                target_tokens: torch.LongTensor = None,
                **kwargs,
                ) -> Dict[str, torch.Tensor]:
        self._reset_variational_dropouts()

        layer_states, state_mask = self._embed_encoder(source_tokens)
        hx, enc_attn_fn, start = self._prepare_dec(layer_states, state_mask.long())
        preds, logits = self._forward_dec(target_tokens, start, enc_attn_fn, hx)

        if not isinstance(source_tokens, torch.Tensor):
            source_tokens = source_tokens['input_ids']  # if the source token is a UserDict for PLM models

        output = {'source': source_tokens, "loss": 0}
        if self.training:
            output['loss'] = self._compute_loss(logits, target_tokens, state_mask)

        if target_tokens is not None:
            total_err = self._compute_metrics(source_tokens, target_tokens, preds, logits)
            target_mask = prepare_input_mask(target_tokens)[1][:, 1:].contiguous().bool()
            self._compute_gini_index(state_mask, target_mask)
            output.update(errno=total_err.tolist())

        output.update(predictions=preds, logits=logits, target=target_tokens)
        return output

    # ========== methods direct called by forward ==============

    def _reset_variational_dropouts(self):
        for m in self.modules():
            if isinstance(m, VariationalDropout):
                m.reset()

    def _prepare_dec(self, layer_states: List[torch.Tensor], state_mask):
        if self._enc_dec_trans_usage == 'consistent':
            layer_states = list(map(self._enc_dec_trans, layer_states))
        hx = self._init_decoder(layer_states, state_mask)
        enc_attn_fn = self._get_enc_attn_fn(layer_states[-1], state_mask)
        default_start = state_mask.new_full((state_mask.size()[0],), fill_value=self._start_id)
        if self.mem is not None:
            del self.mem
        self.mem = SeqCollector()
        return hx, enc_attn_fn, default_start

    def _forward_dec(self, target_tokens, default_start, enc_attn_fn, hx):
        target, target_mask = prepare_input_mask(target_tokens, self._padding_index)

        if self.training and target is None:
            raise ValueError("target must be presented during training")

        # preds: (batch, seq_len)
        # logits: (batch, seq_len, vocab_size)
        # preds, logits = self._forward_dec_loop(start, enc_attn_fn, hx, target, target_mask, runtime_len)
        last_pred = default_start
        num_decoding_steps = self._get_decoding_loop_len(target)
        mem = SeqCollector()
        for timestep in range(num_decoding_steps):
            step_input = self._choose_rnn_input(last_pred, None if target is None else target[:, timestep])
            dec_hist_attn_fn = self._create_runtime_dec_hist_attn_fn(mem, target_mask, timestep)
            cell_out, step_logit, hx = self._forward_dec_loop(step_input, enc_attn_fn, dec_hist_attn_fn, hx)
            # last_pred: (batch,), greedy decoding
            last_pred = torch.argmax(step_logit, dim=-1)
            mem(output=cell_out, logit=step_logit)

        # logits: (batch, seq_len, vocab_size)
        # predictions: (batch, seq_len)
        logits = mem.get_stacked_tensor('logit')
        predictions = logits.argmax(dim=-1)

        return predictions, logits

    def _compute_loss(self, logits, target_tokens, state_mask):
        target, target_mask = prepare_input_mask(target_tokens)
        gold = target[:, 1:].contiguous()       # skip the first <START> token and corresponding mask
        mask = target_mask[:, 1:].contiguous()
        loss = seq_cross_ent(logits, gold, mask, self._training_avg)
        if isinstance(self._embed_encoder, LossModule):
            loss = loss + self._embed_encoder.get_loss()

        if 'hungarian' in self._attn_sup:
            loss = loss + self._hungarian_loss(mask, state_mask)
        return loss

    def _compute_metrics(self, source_tokens, target_tokens, predictions, logits):
        target, target_mask = prepare_input_mask(target_tokens)
        source, source_mask = prepare_input_mask(source_tokens)
        # gold, gold_mask: (batch, max_tgt_len - 1)
        gold = target[:, 1:].contiguous()
        gold_mask = target_mask[:, 1:].contiguous()
        # self.bleu(predictions, gold)
        # batch_xent: (batch,)
        batch_xent = seq_cross_ent(logits, gold, gold_mask, average='none')
        for xent in batch_xent:
            self.ppl(xent)

        total_err = ((predictions != gold) * gold_mask).sum(list(range(gold_mask.ndim))[1:]) > 0
        for instance_err in total_err:
            self.err_rate(instance_err)
        self.item_count += predictions.size()[0]
        for l in source_mask.sum(1):
            self.src_len(l)
        for l in gold_mask.sum(1):
            self.tgt_len(l)
        return total_err

    # ============= methods called by _prepare_dec ===============

    def _init_decoder(self, layer_states, source_mask):
        usage: str = self._enc_dec_trans_usage
        use_first_half: bool = self._embed_encoder.is_bidirectional() and usage not in ('dec_init', 'consistent')

        agg_src = aggregate_layered_state(layer_states, source_mask, self._strategy, use_first_half)
        if usage == 'dec_init':
            agg_src = list(map(self._enc_dec_trans, agg_src))

        dec_states = assign_stacked_states(agg_src, self._decoder.get_layer_num(), self._strategy)

        # init the internal hiddens
        hx = self._decoder.init_hidden_states(dec_states)
        return hx

    def _get_enc_attn_fn(self, source_state, source_mask):
        if 'attn' == self._enc_dec_trans_usage:
            source_state = self._enc_dec_trans(source_state)

        def enc_attn_fn(out):
            if self._enc_attn is None:
                return None
            else:
                return self._enc_attn(out, source_state, source_mask)

        return enc_attn_fn

    # ============= methods called by _forward_dec ===============

    def _get_decoding_loop_len(self, target):
        """compute the number of steps for the decoder loop"""
        if target is not None:
            num_decoding_steps = target.size()[1] - 1
        else:
            num_decoding_steps = self._max_decoding_step
        return num_decoding_steps

    def _choose_rnn_input(self, last_pred, last_gold: Optional):
        """get the input for each loop step"""
        if self.training and np.random.rand(1).item() < self._scheduled_sampling_ratio:
            # use self-predicted tokens for scheduled sampling in training with _scheduled_sampling_ratio
            # step_inputs: (batch,)
            step_inputs = last_pred
        elif not self.training or last_gold is None:
            # either in non-training mode (validation or testing),
            # or no target present (testing)
            # step_inputs: (batch,)
            step_inputs = last_pred
        else:
            # gold choice, normal case in training (scheduler sampling not activated)
            # step_inputs: (batch,)
            step_inputs = last_gold
        return step_inputs

    def _create_runtime_dec_hist_attn_fn(self, mem: SeqCollector, target_mask, timestep: int):
        # build runtime decoder history attention module.
        if self._dec_hist_attn is None:
            dec_hist_attn_fn = lambda out: None

        elif timestep > 0:  # which suggest that output_by_step is not empty
            dec_hist = mem.get_stacked_tensor('output')
            dec_hist_mask = target_mask[:, :timestep] if target_mask is not None else None
            dec_hist_attn_fn = lambda out: self._dec_hist_attn(out, dec_hist, dec_hist_mask)

        else:
            dec_hist_attn_fn = lambda out: out

        return dec_hist_attn_fn

    def _forward_dec_loop(self, step_input, enc_attn_fn, dec_hist_attn_fn, hx: Any):
        """define each step of the decoder loop"""
        inputs_embedding = self._get_step_embedding(step_input)
        cell_inp = self._get_cell_input(inputs_embedding, hx, enc_attn_fn, dec_hist_attn_fn)
        hx, cell_out = self._decoder(cell_inp, hx)
        proj_inp = self._get_proj_input(cell_out, enc_attn_fn, dec_hist_attn_fn)
        step_logit = self._get_step_projection(proj_inp)
        return cell_out, step_logit, hx

    # ------------- methods called within each decoder step --------------

    def _get_step_embedding(self, step_inp):
        # inputs_embedding: (batch, embedding_dim)
        inputs_embedding = self._tgt_embedding(step_inp)
        inputs_embedding = self._tgt_emb_dropout(inputs_embedding)
        return inputs_embedding

    def _get_cell_input(self, inputs_embedding, hx, enc_attn_fn, dec_hist_attn_fn):
        # compute attention context before the output is updated
        # actually only the first layer of decoder is using attention,
        # and only the final output of encoder and decoder history are attended over, not inter-layer states
        prev_output = self._decoder.get_output_state(hx)
        prev_enc_context = enc_attn_fn(prev_output)
        prev_dec_context = dec_hist_attn_fn(prev_output)
        if self._enc_attn is not None and self._attn_sup != 'none':
            self.mem(inp_enc_attn=self._enc_attn.get_latest_attn_weights())  # (b, src_len)

        prev_context = filter_cat([prev_enc_context, prev_dec_context], dim=-1)
        cell_inp = self._dec_inp_attn_comp(prev_context, inputs_embedding)
        return cell_inp

    def _get_proj_input(self, cell_out, enc_attn_fn, dec_hist_attn_fn):
        step_enc_context = enc_attn_fn(cell_out)
        step_dec_context = dec_hist_attn_fn(cell_out)
        if self._enc_attn is not None and self._attn_sup != 'none':
            self.mem(proj_enc_attn=self._enc_attn.get_latest_attn_weights())  # (b, src_len)
        step_context = filter_cat([step_enc_context, step_dec_context], dim=-1)
        proj_inp = self._proj_inp_attn_comp(step_context, cell_out)
        return proj_inp

    def _get_step_projection(self, proj_input):
        proj_input = self._proj_inp_dropout(proj_input)
        step_logit = self._output_projection(proj_input)
        return step_logit

    # ============= methods called by _compute_loss ===============
    # special losses

    def _hungarian_loss(self, tgt_mask, state_mask):
        from ..nl2sql.hungarian_loss import (
            get_hungarian_target_by_weight,
            reverse_probabilities,
            hungarian_l2_loss,
            hungarian_xent_loss,
        )

        attn_weights = self.mem.get_stacked_tensor('proj_enc_attn')     # (b, tgt_len, src_len)
        if self._attn_sup.startswith('rev_'):
            attn_weights = reverse_probabilities(attn_weights, tgt_mask, state_mask)

        target = get_hungarian_target_by_weight(attn_weights)

        if self._attn_sup == 'hungarian_reg':
            return hungarian_l2_loss(attn_weights, target, tgt_mask, state_mask, regularized=True)

        elif self._attn_sup == 'hungarian_sup':
            return hungarian_l2_loss(attn_weights, target, tgt_mask, state_mask)

        elif self._attn_sup == 'hungarian_xent':
            return hungarian_xent_loss(attn_weights, target, tgt_mask, state_mask)

        elif self._attn_sup == 'hungarian_reg_xent':
            return hungarian_xent_loss(attn_weights, target, tgt_mask, state_mask, regularized=True)

        elif self._attn_sup == 'rev_hungarian_reg_xent':
            return hungarian_xent_loss(attn_weights, target, tgt_mask, state_mask, regularized=True)

        elif self._attn_sup == 'rev_hungarian_xent':
            return hungarian_xent_loss(attn_weights, target, tgt_mask, state_mask)

        else:
            return 0

    def _compute_gini_index(self, state_mask, tgt_mask):
        if self.training or self._attn_sup == 'none':
            return

        from utils.gini import slow_gini_index
        from allennlp.nn.util import masked_mean

        attn_weight = self.mem.get_stacked_tensor('proj_enc_attn')  # (b, tgt_len, src_len)
        if attn_weight is None:
            logging.getLogger(self.__class__.__name__).warning(
                "attention not saved and thus unfetchable"
            )
            return

        gini = slow_gini_index(attn_weight, state_mask.sum(dim=-1)) # (b, tgt_len)
        sparsities = masked_mean(gini, tgt_mask, dim=-1)    # (b,)
        for v in sparsities:
            self.sparsity(v)
