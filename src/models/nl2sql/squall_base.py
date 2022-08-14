from itertools import product, chain
from typing import Dict, Any, Tuple, List

import torch
import re
from allennlp.training.metrics import Average
from allennlp.nn.util import masked_log_softmax
from fuzzywuzzy import fuzz
from torch import nn
from trialbot.data import NSVocabulary

from models.interfaces.attention import Attention as IAttn
from models.interfaces.encoder import Encoder
from models.interfaces.unified_rnn import RNNStack
from models.nl2sql.p_tuning_v2 import PTuningV2Prompt
from utils.nn import aggregate_layered_state, assign_stacked_states, masked_reducing_gather, compact_mask_select
from utils.seq_collector import SeqCollector


class SquallBaseParser(nn.Module):
    def __init__(self,
                 # modules
                 plm_model,
                 word_plm2hidden: nn.Module,
                 col_plm2hidden: nn.Module,
                 word_enc: Encoder,
                 col_enc: Encoder,

                 word_col_attn: IAttn,
                 col_word_attn: IAttn,

                 kwd_embedding: nn.Module,
                 col2input: nn.Module,
                 span2input: nn.Module,
                 decoder: RNNStack,

                 sql_word_attn: IAttn,
                 sql_col_attn: IAttn,
                 sql_type: nn.Module,
                 sql_keyword: nn.Module,
                 sql_col_type: nn.Module,
                 sql_col_copy: IAttn,
                 sql_span_begin: IAttn,
                 sql_span_end: IAttn,

                 start_token: str,
                 end_token: str,
                 vocab: NSVocabulary,
                 ns_keyword: str,
                 ns_coltype: str,

                 # optional modules
                 aux_col: IAttn = None,
                 p_tuning: PTuningV2Prompt = None,

                 # configs
                 src_type_keys: tuple = ('pad', 'special', 'word', 'word_pivot', 'column', 'col_pivot'),
                 tgt_type_keys: tuple = ('pad', 'keyword', 'column', 'literal_string', 'literal_number'),
                 decoder_init_strategy: str = "zero_all",
                 ):
        super().__init__()
        self.pretrained_model = plm_model
        self.word_plm2hidden = word_plm2hidden
        self.col_plm2hidden = col_plm2hidden
        self.word_enc = word_enc
        self.col_enc = col_enc

        self.aux_word_col = aux_col

        self.word_col_attn = word_col_attn  # encoder use
        self.col_word_attn = col_word_attn
        self.sql_word_attn = sql_word_attn  # decoder use
        self.sql_col_attn = sql_col_attn

        self.sql_type = sql_type
        self.sql_keyword = sql_keyword
        self.sql_col_type = sql_col_type
        self.sql_col_copy = sql_col_copy
        self.sql_span_begin = sql_span_begin
        self.sql_span_end = sql_span_end

        self.sql_keyword_embed = kwd_embedding  # requires output being hidden
        self.col2input = col2input
        self.span2input = span2input
        self.decoder = decoder

        self.p_tuning = p_tuning
        if self.p_tuning is not None:
            self.pretrained_model: torch.nn.Module
            self.pretrained_model.requires_grad_(False)

        self._strategy = decoder_init_strategy
        self.start_token = start_token
        self.end_token = end_token
        self.vocab = vocab
        self.ns_keyword = ns_keyword
        self.ns_coltype = ns_coltype

        self.stype_map = dict(zip(src_type_keys, range(len(src_type_keys))))
        self.ttype_map = dict(zip(tgt_type_keys, range(len(tgt_type_keys))))

        self.step_prediction_keys = ('ttype', 'keyword', 'col_type', 'col_copy', 'span_begin', 'span_end')
        # the decoder-to-column attention is also meaningful but not used for training and evaluation
        self.step_attn_keys = ('word_attn',)

        # metrics
        self.acc = Average()
        self.word_num = Average()
        self.col_num = Average()
        self.tgt_len = Average()
        self.mean_loss = Average()
        self.dec_acc_pred = Average()
        self.item_count = 0
        self.match_stats = None

    def get_metrics(self, reset=False):
        metric = {"ACC_EM": round(self.acc.get_metric(reset), 6),
                  "ACCPred": round(self.dec_acc_pred.get_metric(reset), 6),
                  "WORDS": round(self.word_num.get_metric(reset), 2),
                  "COLUMNS": round(self.col_num.get_metric(reset), 2),
                  "TLEN": round(self.tgt_len.get_metric(reset), 2),
                  "AVG_LOSS": round(self.mean_loss.get_metric(reset), 6),
                  "COUNT": self.item_count,
                  "TaskAcc": [round(n / d, 4) if d > 0 else 'nan' for n, d in self.match_stats],
                  }
        if reset:
            self.item_count = 0
            self.match_stats = None
        return metric

    def forward(self,
                # encoder inputs
                src_ids, src_types: torch.Tensor, src_plm_type_ids,
                # alignments between each word/sql, word/col, and sql/col pair
                align_ws_word, align_ws_sql, align_wc_word, align_wc_col, align_sc_sql, align_sc_col,
                # decoder supervisions,
                tgt_type: torch.Tensor, tgt_keyword, tgt_col_type, tgt_col_id, tgt_literal_begin, tgt_literal_end,
                # extension for future uses
                **kwargs
                ):
        word_states, word_mask, col_states, col_mask = self.encode(src_ids, src_types, src_plm_type_ids)
        hx = self.init_decoder(word_states, word_mask)

        mem = SeqCollector()
        num_decoding_steps = tgt_type.size(1) - 1 if tgt_type is not None else 100    # baseline maximum
        for step in range(num_decoding_steps):
            if step == 0:
                step_emb = self.get_first_step_embedding(word_mask.size(0), word_mask.device)
            elif self.training:
                step_input = self.get_gold_step(step, tgt_type, tgt_keyword, tgt_col_id,
                                                tgt_literal_begin, tgt_literal_end)
                step_emb = self.get_step_embedding(word_states, col_states, *step_input)
            else:
                step_input = self.get_pred_step(mem)
                step_emb = self.get_step_embedding(word_states, col_states, *step_input)

            hx, out = self.decoder(step_emb, hx)

            if self.training:
                mem(step_out=out)
            else:
                word_ctx, col_ctx = self.get_enc_attn(out, word_states, word_mask, col_states, col_mask)
                hvec = torch.cat([out, word_ctx, col_ctx], dim=-1)   # (b, hidden * 3)
                self.step_predictions(hvec, word_states, word_mask, col_states, col_mask, mem)

        if self.training:
            all_step_outs = mem.get_stacked_tensor('step_out')
            word_ctx, col_ctx = self.get_enc_attn(all_step_outs, word_states, word_mask, col_states, col_mask)
            hvec = torch.cat([all_step_outs, word_ctx, col_ctx], dim=-1)    # (b, len, hidden * 5)
            self.step_predictions(hvec, word_states, word_mask, col_states, col_mask, mem)

        logprob_dict = self.get_logprobs(mem, col_type_mask=kwargs.get('col_type_mask'))
        # following the order in self.step_prediction_keys for convenience
        gold_predictions = (tgt_type[:, 1:], tgt_keyword[:, 1:], tgt_col_type[:, 1:],
                            tgt_col_id[:, 1:], tgt_literal_begin[:, 1:], tgt_literal_end[:, 1:])
        losses, matches, match_stat = self.get_prediction_losses(logprob_dict, *gold_predictions)
        losses.extend(self.get_alignment_loss(mem, align_ws_word, align_ws_sql, align_wc_word, align_wc_col,
                                              word_mask, col_mask, tgt_type[:, 1:] != self.ttype_map['pad']))
        if self.aux_word_col is not None:
            aux_loss = self.get_aux_loss(word_states, col_states, col_mask, align_wc_word, align_wc_col)
            losses.append(0.2 * aux_loss)
        final_loss = sum(losses)

        self.compute_metrics(word_mask, col_mask, matches, match_stat, tgt_type[:, 1:], final_loss)

        output = {"src_id": src_ids, "src_type": src_types, "tgt_type": tgt_type,
                  "tgt_keyword": tgt_keyword, "tgt_col_type": tgt_col_type, "tgt_col_id": tgt_col_id,
                  "tgt_literal_begin": tgt_literal_begin, "tgt_literal_end": tgt_literal_end,
                  "loss": final_loss,
                  }

        if not self.training:
            # during training the logits are only evaluated
            predictions = self.decode_logprob(logprob_dict)
            pred_queries, _ = self.realisation(predictions, kwargs.get('nl_toks'), kwargs.get('tbl_cells'))
            self.compute_realisation_metrics(pred_queries, kwargs.get('sql_toks'))
            output.update(pred_queries=pred_queries, target=kwargs.get('sql_toks'), nl_toks=kwargs.get('nl_toks'))

        return output

    # ------------ encoding steps ------------

    def encode(self, src_ids, src_types: torch.Tensor, plm_type_ids):
        state_mask: torch.Tensor = (src_types != self.stype_map['pad'])
        past_key_values = None
        if self.p_tuning is not None:
            nbatch = state_mask.size(0)
            past_key_values = self.p_tuning.forward(nbatch)
            prefix_mask = state_mask.new_ones((nbatch, self.p_tuning.prefix_len))
            state_mask = torch.cat([prefix_mask, state_mask], dim=-1)

        enc_out = self.pretrained_model(input_ids=src_ids,
                                        token_type_ids=plm_type_ids,
                                        attention_mask=state_mask.float(),
                                        past_key_values=past_key_values)
        enc_states = enc_out.last_hidden_state

        def st_is(k: str):
            return src_types == self.stype_map[k]

        word_states, word_state_mask = compact_mask_select(enc_states, st_is('word_pivot'))
        col_states, col_state_mask = compact_mask_select(enc_states, st_is('col_pivot'))
        word_states = self.word_plm2hidden(word_states)
        col_states = self.col_plm2hidden(col_states)

        # ctx: (b, len, hid)
        # attn: (b, input_len, num_heads=1, attend_len)
        col_ctx = self.word_col_attn.forward(word_states, col_states, col_state_mask)
        word_ctx = self.col_word_attn.forward(col_states, word_states, word_state_mask)

        word_states = self.word_enc.forward(torch.cat([word_states, col_ctx], dim=-1), word_state_mask)
        col_states = self.col_enc.forward(torch.cat([col_states, word_ctx], dim=-1), col_state_mask)

        return word_states, word_state_mask, col_states, col_state_mask

    def init_decoder(self, states, state_mask):
        agg_src = aggregate_layered_state([states], state_mask, self._strategy)
        dec_states = assign_stacked_states(agg_src, self.decoder.get_layer_num(), self._strategy)
        # init the internal hiddens
        hx, _ = self.decoder.init_hidden_states(dec_states)
        return hx

    # ------------ decoder looping functions  --------------
    # preparing decoder inputs, computing attentions, and do predictions and save logits in memory

    def get_first_step_embedding(self, nbatch: int, device=None):
        start = torch.full((nbatch,),
                           fill_value=self.vocab.get_token_index(self.start_token, self.ns_keyword),
                           device=device)
        start_emb = self.sql_keyword_embed(start)   # (batch, hid)
        return start_emb

    def get_gold_step(self, step: int, tgt_type, keywords, col_id, span_begin, span_end):
        return tgt_type[:, step], keywords[:, step], col_id[:, step], span_begin[:, step], span_end[:, step]

    def get_pred_step(self, mem: SeqCollector):
        # col-type is not required for prediction, so no mask is needed
        output = self.get_logprobs(mem, -1)
        k_type, k_keyword, _, k_col, k_begin, k_end = self.step_prediction_keys
        step_inp = tuple(map(lambda k: output[k].argmax(dim=-1), (k_type, k_keyword, k_col, k_begin, k_end)))
        return step_inp

    def get_step_embedding(self, word_states, col_states, ttype, keywords, col_id, span_begin, span_end):
        nbatch = ttype.size(0)
        step_input_type = ttype.unsqueeze(-1)   # (b, 1)
        emb_inp = self.sql_keyword_embed(keywords)
        col_inp = self.col2input(col_states[range(nbatch), col_id])
        span_begin_state = word_states[range(nbatch), span_begin]
        span_end_state = word_states[range(nbatch), span_end]
        len1_span_inp = self.span2input(torch.cat([span_begin_state, span_begin_state], dim=-1))
        len2_span_inp = self.span2input(torch.cat([span_begin_state, span_end_state], dim=-1))

        is_pad = step_input_type == self.ttype_map['pad']
        is_key = step_input_type == self.ttype_map['keyword']
        is_col = step_input_type == self.ttype_map['column']
        is_str = step_input_type == self.ttype_map['literal_string']
        is_num = step_input_type == self.ttype_map['literal_number']

        # (batch, hid)
        inp = (is_pad * emb_inp + is_key * emb_inp + is_col * col_inp
               + is_num * len1_span_inp + is_str * len2_span_inp)
        return inp

    def get_enc_attn(self, step_out, word_states, word_state_mask, col_states, col_state_mask):
        # step_out: (b, hid)
        # *_mask: (b, n)
        word_context = self.sql_word_attn.forward(step_out, word_states, word_state_mask)
        col_context = self.sql_col_attn.forward(step_out, col_states, col_state_mask)
        # attn_weights for word_context will be added in self.step_predictions for training supervision
        # and thus will not be returned here.
        return word_context, col_context

    def step_predictions(self, hvec, word_state, word_mask, col_state, col_mask, mem):
        # classification-based predictions
        ttype_logit = self.sql_type(hvec)  # (b, #types)
        keyword_logit = self.sql_keyword(hvec)  # (b, #keywords)
        col_type_logit = self.sql_col_type(hvec)  # (b, #coltypes)

        # attention-based predictions
        self.sql_col_copy.forward(hvec, col_state, col_mask)
        col_copy_prob = self.sql_col_copy.get_latest_attn_weights()  # (b, n)
        self.sql_span_begin.forward(hvec, word_state, word_mask)
        span_begin_prob = self.sql_span_begin.get_latest_attn_weights()  # (b, n)
        self.sql_span_end.forward(hvec, word_state, word_mask)
        span_end_prob = self.sql_span_end.get_latest_attn_weights()  # (b, n)
        # for encoder attention training supervision
        word_attn_prob = self.sql_word_attn.get_latest_attn_weights()    # (b, n)

        mem(**dict(zip(self.step_prediction_keys, (ttype_logit, keyword_logit, col_type_logit,
                                                   col_copy_prob, span_begin_prob, span_end_prob))))
        mem(**{self.step_attn_keys[0]: word_attn_prob})

    # ------------- post-decoding steps -------------------
    # for loss and metric computations

    def get_prediction_losses(self, logprob_dict,
                              gold_type: torch.LongTensor, gold_keyword, gold_col_type,
                              gold_col_id, gold_span_begin, gold_span_end):
        losses, matches, match_stat = [], [], []

        def _add_loss(logprob, target, mask: torch.BoolTensor):
            loss_matrix = torch.gather(logprob, index=target.unsqueeze(-1), dim=-1).squeeze(-1)
            losses.append(- (loss_matrix * mask).sum() / mask.size(0))
            correct = logprob.argmax(dim=-1) == target
            matches.append(torch.bitwise_or(~mask, correct).all(dim=-1))
            nominator = torch.bitwise_and(mask, correct).sum().item()
            denominator = mask.sum().item()
            match_stat.append([nominator, denominator])

        def is_type(k: str):
            return gold_type == self.ttype_map[k]

        ks = self.step_prediction_keys

        # classification logprobs: (b, tgt_len, #V)
        _add_loss(logprob_dict[ks[0]], gold_type, ~is_type('pad'))
        _add_loss(logprob_dict[ks[1]], gold_keyword, is_type('keyword'))
        _add_loss(logprob_dict[ks[2]], gold_col_type, is_type('column'))

        # copy logprobs, with gold selection labels: (b, tgt_len, src_len)
        _add_loss(logprob_dict[ks[3]], gold_col_id, is_type('column'))
        span_begin_log = logprob_dict[ks[4]]
        _add_loss(span_begin_log, gold_span_begin, is_type('literal_number'))
        _add_loss(span_begin_log, gold_span_begin, is_type('literal_string'))
        _add_loss(logprob_dict[ks[5]], gold_span_end, is_type('literal_string'))

        return losses, matches, match_stat

    def get_alignment_loss(self, mem: SeqCollector, ws_word, ws_sql, wc_word, wc_col,
                           word_mask, col_mask, sql_mask):
        # decoder to word attention
        if self.training:
            s2w_attn = mem[self.step_attn_keys[0]][-1]
        else:
            s2w_attn = mem.get_stacked_tensor(self.step_attn_keys[0])

        # encoder attention
        w2c_attn = self.word_col_attn.get_latest_attn_weights()  # (b, num_words, num_cols)
        c2w_attn = self.col_word_attn.get_latest_attn_weights()  # (b, num_cols, num_words)

        s2w_loss = self._get_attn_sup_loss(s2w_attn, ws_sql, ws_word, sql_mask, word_mask)
        w2c_loss = self._get_attn_sup_loss(w2c_attn, wc_word, wc_col, word_mask, col_mask)
        c2w_loss = self._get_attn_sup_loss(c2w_attn, wc_col, wc_word, col_mask, word_mask)

        losses = [s2w_loss, w2c_loss, c2w_loss]
        return losses

    def _get_attn_sup_loss(self, attn_weights, align_vec, align_mat, vec_mask, mat_mask):
        """
        Attention L2 loss under different supervision settings.

        :param attn_weights: (b, #vec, #mat)
        :param align_vec:   (b, #align), padded by -1
        :param align_mat:   (b, #align), padded by -1
        :param vec_mask:    (b, #vec)
        :param mat_mask:    (b, #mat)
        :return: (), a torch scalar for the L2 loss
        """
        nbatch, dev = attn_weights.size(0), attn_weights.device
        batch = torch.arange(nbatch, device=dev).unsqueeze(-1)          # (b, 1)
        pad_mask = vec_mask.unsqueeze(-1) * mat_mask.unsqueeze(-2)      # (b, #vec, #mat)
        align_mask = align_vec >= 0

        attn_target = torch.zeros_like(attn_weights)        # (b, #vec, #mat)
        dup_num = (align_vec.unsqueeze(-1) == align_vec.unsqueeze(-2)).sum(-1)  # (b, #align)
        # use align_mask to reset invalid alignments (-1 when padding, for example).
        # since the target is defaulted to 0, setting invalid alignments to 0 is just fine.
        attn_target[batch, align_vec, align_mat] = dup_num.reciprocal() * align_mask

        # use attn_mask to reset loss at padding locations
        raw_l2_loss = (attn_weights - attn_target) ** 2 / 2 * pad_mask
        # following the baseline choice,
        # attentions related to unaligned vec or unaligned mat must also be trained towards 0
        return raw_l2_loss.sum(dim=(1, 2)).mean()

    def get_aux_loss(self, word_states, col_states, col_mask, align_word, align_col):
        # NLL loss for the the attention module
        _ = self.aux_word_col.forward(word_states, col_states, col_mask)
        col_attn = self.aux_word_col.get_latest_attn_weights()  # (b, word_len, col_len)

        batch = torch.arange(align_word.size()[0], device=align_word.device).unsqueeze(-1)
        col_probs = col_attn[batch, align_word, align_col]   # (batch, num_alignments)
        align_mask = align_word >= 0
        col_logprobs = col_probs.clamp(min=1e-20).log() * align_mask     # (batch, num_alignments)
        aux_loss = - col_logprobs.sum(-1).mean()
        return aux_loss

    def compute_metrics(self, word_pivot, col_pivot, matches, match_stat, tgt_type, final_loss):
        for n in word_pivot.sum(-1):
            self.word_num(n)
        for n in col_pivot.sum(-1):
            self.col_num(n)
        for m in torch.stack(matches, dim=1).all(dim=1).long():
            self.acc(m)
        for n in (tgt_type == self.ttype_map['pad']).sum(-1):
            self.tgt_len(n)
        self.item_count += word_pivot.size()[0]
        self.mean_loss(final_loss)

        if self.match_stats is None:
            self.match_stats = match_stat
        else:
            for i, (n, d) in enumerate(match_stat):
                self.match_stats[i][0] += n
                self.match_stats[i][1] += d

    def decode_logprob(self, logprob_dict: dict):
        return tuple(logprob_dict[k].argmax(dim=-1) for k in self.step_prediction_keys)

    def realisation(self, predictions: tuple, nl_toks: List[List[str]], tbl_cells: List[set]):
        # all predictions: (batch, tgt_len)
        # ttype, tkwd, coltype, col, span_begin, span_end = predictions
        nbatch, nsteps = predictions[0].size()
        type_str_map = ('-', 'Keyword', 'Column', 'Literal.String', 'Literal.Number')   # used by the baseline evaluator
        batch_query, batch_types = [], []
        for i in range(nbatch):
            query, types = [], []
            for j in range(nsteps):
                ttype, kwd, coltype, col, span_begin, span_end = tuple(x[i, j].item() for x in predictions)
                if ttype == self.ttype_map['keyword']:
                    tok = self.vocab.get_token_from_index(kwd, self.ns_keyword)
                    if tok == self.end_token:
                        break   # the only break: types and query list will stop increasing when encounter EOS
                    types.append(type_str_map[ttype])
                    query.append(tok)

                elif ttype == self.ttype_map['column']:
                    types.append(type_str_map[ttype])
                    coltype_str = self.vocab.get_token_from_index(coltype, self.ns_coltype)
                    coltype_suffix = "" if coltype_str == "none" else "_" + coltype_str
                    col_str = f"c{col + 1}{coltype_suffix}"
                    query.append(col_str)

                elif ttype == self.ttype_map['literal_string']:
                    types.append(type_str_map[ttype])
                    assert span_end >= span_begin
                    substring = ' '.join(nl_toks[i][span_begin:span_end + 1])
                    literal = self._process_with_string_patterns(query, substring, tbl_cells[i], types)
                    query.append("{}".format(repr(literal)))

                elif ttype == self.ttype_map['literal_number']:
                    types.append(type_str_map[ttype])
                    query.append("{}".format(parse_number(nl_toks[i][span_begin])))

                else:
                    pass

            batch_query.append(query)
            batch_types.append(types)

        return batch_query, batch_types

    def compute_realisation_metrics(self, pred_queries, target_sqls: List[List[str]]):
        for pred_toks, sql_toks in zip(pred_queries, target_sqls):
            pred_query = " ".join(pred_toks)
            sql = " ".join(sql_toks)
            self.dec_acc_pred(1. if pred_query == sql else 0.)

    @staticmethod
    def _process_with_string_patterns(query, substring, tbl_cells, types):
        # the handcrafted pattens strictly follow the baseline code
        if len(query) >= 2 and query[-1] == '=' and types[-3] == 'Column':
            # colstr may get updated if no exact col as query[-2] is matched
            colstr, literal = best_match(tbl_cells, substring, query[-2])
        else:
            colstr, literal = best_match(tbl_cells, substring)
        # new query token has been added, so the indices are moved left by 1
        if len(query) >= 2 and query[-1] == "=" and types[-2] == "Column":
            query[-2] = colstr
        if len(query) >= 3 and query[-1] == "(" and query[-2] == "in" and types[-3] == "Column":
            query[-3] = colstr

        return literal

    # ------------ util function ---------------
    # During training, the decoder uses gold inputs and collects step outputs at each looping step,
    # and in the end build the logprobs (1) for loss and metric computations.
    #
    # During evaluation, the dedocer uses last predicted outputs as inputs (2) at each looping step,
    # but still collect the step outputs to build the logprobs (3) after the decoding loops.
    # And the metrics (even losses) computations will be dependent on the logprobs.
    #
    # For comparisons with the baseline, we also reported the accuracies of not only the model predictions
    # but also the raw SQL string. To convert the model predictions into target SQLs, we further use
    # the logprob (4) produced from (3) and doing some post processings.
    #
    # For all the cases (1), (2), (3), and (4), the following get_logprobs method is necessary.
    #

    def get_logprobs(self, mem: SeqCollector, step: int = None, col_type_mask: torch.LongTensor = None) -> dict:
        """
        Output the logprobs for each prediction key.

        :param mem: decoder memory to save outputs of each step
        :param col_type_mask: (b, #col-nums, V)
        :param step: (b, #col-nums, V)
        """
        def kget(key: str): return mem.get_stacked_tensor(key) if step is None else mem[key][step]

        # no decoding mask will be applied during training, simply logprob or log is fine
        if self.training:
            keys_and_funcs = chain(product(self.step_prediction_keys[:3], [logprob]),
                                   product(self.step_prediction_keys[3:], [log]))
            # for training, we use a one-turn-prediction instead of applying argmax step by step
            output = {k: foo(mem[k][-1]) for k, foo in keys_and_funcs}
            return output

        ks = self.step_prediction_keys
        output = {k: logprob(kget(k)) for k in ks[:2]}  # first two keys doesn't have dynamic decoding masks
        _, _, k_col_t, k_col, k_begin, k_end = ks   # keys for col-type, col-copy, span-begin, span-end

        # for col_copy no mask is present
        col_logp = log(kget(k_col))
        output[k_col] = col_logp    # (b, *, #col-id)

        # for col-type we may need to build a dynamic mask
        col_id = col_logp.argmax(dim=-1)   # (b, *)
        dynamic_mask = None
        if col_type_mask is not None:
            batch = torch.arange(col_id.size(0), device=col_id.device)  # (b,)
            if step is None:
                batch = batch.unsqueeze(-1)     # (b, 1)
            dynamic_mask = col_type_mask[batch, col_id]     # (b, *, #col-type)
        output[k_col_t] = logprob(kget(k_col_t), dynamic_mask)

        # for span-begin, no mask is needed
        span_begin_logp = log(kget(k_begin))
        output[k_begin] = span_begin_logp    # (b, *, src_len)

        # for span-end, the mask must be built to clean probabilities before the span begin token.
        span_end_prob = kget(k_end)  # (b, *, src_len)
        nwords = span_end_prob.size()[-1]
        span_end_mask = torch.ones_like(span_end_prob).reshape(-1, nwords)  # (b[* tgt_len], src_len)
        span_begin = span_begin_logp.argmax(dim=-1).view(-1)     # (b[* tgt_len],)
        for i, begin in enumerate(span_begin):
            span_end_mask[i, :begin] = 0    # exclude tokens proceeding the begin token, for the whole batch
        span_end_mask = span_end_mask.reshape_as(span_end_prob)
        span_end_logp = log(span_end_prob, span_end_mask)
        output[k_end] = span_end_logp
        return output


def logprob(logits, mask=None):
    if mask is None:
        return torch.log_softmax(logits, dim=-1)
    return masked_log_softmax(logits, mask, dim=-1)


def log(prob, mask=None):
    if mask is None:
        return prob.clamp(min=1e-20).log()
    else:
        scale = (prob * mask).sum(dim=-1, keepdims=True).clamp(min=1e-20)
        return ((prob * mask) / scale).clamp(min=1e-20).log()


def best_match(candidates, query, col=None):
    return max(candidates, key=lambda x: (fuzz.ratio(x[1], query), col==x[0]))


def parse_number(s):
    if s in NUM_MAPPING:
        return NUM_MAPPING[s]

    s = s.replace(',', '')
    # https://stackoverflow.com/questions/4289331/python-extract-numbers-from-a-string
    ret = re.findall(r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", s)
    if len(ret) > 0:
        return ret[0]

    return None


NUM_MAPPING = {
    'half': 0.5,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'eleven': 11,
    'twelve': 12,
    'twenty': 20,
    'thirty': 30,
    'once': 1,
    'twice': 2,
    'first': 1,
    'second': 2,
    'third': 3,
    'fourth': 4,
    'fifth': 5,
    'sixth': 6,
    'seventh': 7,
    'eighth': 8,
    'ninth': 9,
    'tenth': 10,
    'hundred': 100,
    'thousand': 1000,
    'million': 1000000,
    'jan': 1,
    'feb': 2,
    'mar': 3,
    'apr': 4,
    'may': 5,
    'jun': 6,
    'jul': 7,
    'aug': 8,
    'sep': 9,
    'oct': 10,
    'nov': 11,
    'dec': 12,
    'january': 1,
    'february': 2,
    'march': 3,
    'april': 4,
    'june': 6,
    'july': 7,
    'august': 8,
    'september': 9,
    'october': 10,
    'november': 11,
    'december': 12,
}

