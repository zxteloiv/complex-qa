from typing import List

import torch
import torch.nn.functional as F
from allennlp.nn.util import masked_softmax, masked_log_softmax
from allennlp.training.metrics import Average
from torch import nn

from models.interfaces.encoder import StackEncoder
from models.modules.attentions.adaptive_attention import AdaptiveGeneralAttention as Attn, AdaptiveAllenLogits as AllenLogits
from models.base_s2s.model_factory import EncoderStackMixin
from allennlp.modules.matrix_attention import BilinearMatrixAttention

from utils.nn import compact_mask_select
from models.nl2sql.hungarian_loss import get_hungarian_reg_loss


class SQLova(nn.Module):
    @classmethod
    def build(cls, p, vocab):
        from transformers import AutoModel
        plm_model = AutoModel.from_pretrained(p.plm_model)
        plm_out_size = plm_model.config.hidden_size
        hid_sz = p.hidden_sz

        WIKISQL_AGG_OPS = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
        WIKISQL_COND_OPS = ['=', '>', '<', 'OP']    # nobody knows what the OP is

        def _get_bilinear(inp_sz, hid_sz=None, is_sparse: bool = False):
            hid_sz = hid_sz or inp_sz
            attn = Attn(AllenLogits(BilinearMatrixAttention(inp_sz, hid_sz)), is_sparse=is_sparse)
            return attn

        inp_sz = plm_out_size * 2

        model = cls(
            plm_model=plm_model,
            mod_select_column=SelectColumn(
                WordColBiEnc.build(inp_sz, hid_sz, p.num_layers, p.dropout), _get_bilinear(hid_sz), hid_sz
            ),
            mod_select_agg=SelectAgg(
                WordColBiEnc.build(inp_sz, hid_sz, p.num_layers, p.dropout), _get_bilinear(hid_sz), hid_sz,
                len(WIKISQL_AGG_OPS),
            ),
            mod_where_number=WhereNumber(
                WordColBiEnc.build(inp_sz, hid_sz, p.num_layers, p.dropout), hid_sz, p.num_layers,
                p.max_conds,
            ),
            mod_where_column=WhereColumn(
                WordColBiEnc.build(inp_sz, hid_sz, p.num_layers, p.dropout), _get_bilinear(hid_sz), hid_sz,
            ),
            mod_where_operator=WhereOps(
                WordColBiEnc.build(inp_sz, hid_sz, p.num_layers, p.dropout), _get_bilinear(hid_sz), hid_sz,
                len(WIKISQL_COND_OPS),
            ),
            mod_where_value=WhereValue(
                WordColBiEnc.build(inp_sz, hid_sz, p.num_layers, p.dropout), _get_bilinear(hid_sz), hid_sz,
                len(WIKISQL_COND_OPS),
                begin_scorer=_get_bilinear(hid_sz * 3, hid_sz),
                end_scorer=_get_bilinear(hid_sz * 3, hid_sz),
            ),
            use_metric_adaptive_losses=getattr(p, 'use_metric_adaptive_losses', False),
            use_hungarian_loss=getattr(p, 'use_hungarian_loss', 'none'),
        )
        return model

    def __init__(self,
                 # encoding modules
                 plm_model: nn.Module,
                 mod_select_column: 'SelectColumn',
                 mod_select_agg: 'SelectAgg',
                 mod_where_number: 'WhereNumber',
                 mod_where_column: 'WhereColumn',
                 mod_where_operator: 'WhereOps',
                 mod_where_value: 'WhereValue',
                 use_metric_adaptive_losses: bool = False,
                 use_hungarian_loss: str = 'none',  # none, partial, and full
                 ):
        super().__init__()
        self.plm_model = plm_model
        self.src_type_keys: tuple = ('pad', 'special', 'word', 'word_pivot', 'column', 'col_pivot')

        self.stype_map = dict(zip(self.src_type_keys, range(len(self.src_type_keys))))
        # all the multi-task (classification) modules accept the inputs of
        # words: (b, max_words, dim), and column(headers): (b, num_headers, dim)
        # and their corresponding masks (b, max_words), and (b, num_headers)
        self.sel_col = mod_select_column
        self.sel_agg = mod_select_agg
        self.wh_num = mod_where_number
        self.wh_col = mod_where_column
        self.wh_op = mod_where_operator
        self.wh_val = mod_where_value

        self.item_count = 0
        self.mean_loss = Average()
        self.word_num = Average()
        self.col_num = Average()
        self.acc = Average()
        self.acc_select = Average()
        self.acc_agg = Average()
        self.acc_num = Average()
        self.acc_cols = Average()
        self.acc_ops = Average()
        self.acc_begin = Average()
        self.acc_end = Average()

        self.use_metric_adaptive_losses = use_metric_adaptive_losses
        self.use_hungarian_loss = use_hungarian_loss

    def get_metrics(self, reset=False):
        metric = {
            "ACC": round(self.acc.get_metric(reset), 6),
            "WORDS": round(self.word_num.get_metric(reset), 2),
            "COLUMNS": round(self.col_num.get_metric(reset), 2),
            "AVG_LOSS": round(self.mean_loss.get_metric(reset), 6),
            "COUNT": self.item_count,
            "MODULE_ACC": {
                "select": round(self.acc_select.get_metric(reset), 4),
                "agg": round(self.acc_agg.get_metric(reset), 4),
                "cond_num": round(self.acc_num.get_metric(reset), 4),
                "cond_cols": round(self.acc_cols.get_metric(reset), 4),
                "cond_ops": round(self.acc_ops.get_metric(reset), 4),
                "span_begin": round(self.acc_begin.get_metric(reset), 4),
                "span_end": round(self.acc_end.get_metric(reset), 4),
            }
        }
        if reset:
            self.item_count = 0
        return metric

    def forward(self,
                # encoder params
                src_ids, src_types, src_plm_type_ids,   # encoder params
                # batch-wise oracle
                select, agg, where_num,     # (b,)
                # column-wise oracle
                where_cols, where_ops, where_begin, where_end,  # (b, Nc), padded by -1, Nc is the number of columns
                **kwargs    # in case of irrelevant input
                ):
        words, word_mask, cols, col_mask = self.encode(src_ids, src_types, src_plm_type_ids)
        sel_logp = self.sel_col(words, cols, word_mask, col_mask)   # (b, Nc)
        agg_logp = self.sel_agg(words, cols, word_mask, col_mask)   # (b, Nc, num_aggs)
        cond_num_logp = self.wh_num(words, cols, word_mask, col_mask)    # (b, 1 + max_conds)
        col_prob = self.wh_col(words, cols, word_mask, col_mask)    # (b, Nc)
        ops_logp = self.wh_op(words, cols, word_mask, col_mask)     # (b, Nc, num_ops)
        if not self.training:
            ops_inp = F.one_hot(ops_logp[:, :, :-1].argmax(dim=-1), num_classes=ops_logp.size()[-1]).float()
        else:
            ops_inp = F.one_hot(where_ops * (where_ops >= 0), num_classes=ops_logp.size()[-1]).float()
        begin_prob, end_prob = self.wh_val(words, cols, word_mask, col_mask, ops_inp)    # (b, Nc, Nw)

        loss = self.get_loss(sel_logp, agg_logp, cond_num_logp, col_prob, ops_logp, begin_prob, end_prob,
                             select, agg, where_num, where_cols, where_ops, where_begin, where_end)
        if self.use_hungarian_loss != 'none':
            alignment_loss = self.get_alignment_losses(word_mask, col_mask)
            loss = loss + alignment_loss

        self.compute_metrics(loss, word_mask, col_mask,
                             sel_logp, agg_logp, cond_num_logp, col_prob, ops_logp, begin_prob, end_prob,
                             select, agg, where_num, where_cols, where_ops, where_begin, where_end)

        output = {'loss': loss}
        return output

    def encode(self, src_ids, src_types: torch.LongTensor, plm_type_ids):
        state_mask: torch.Tensor = (src_types != self.stype_map['pad'])
        enc_out = self.plm_model(input_ids=src_ids,
                                 token_type_ids=plm_type_ids,
                                 attention_mask=state_mask.float(),
                                 output_hidden_states=True)

        layer_n2, layer_n1 = enc_out.hidden_states[-2:]  # the negative 2 and negative 1, i.e., the last two layers

        def st_is(k: str) -> torch.LongTensor: return src_types == self.stype_map[k]    # noqa

        word_n2, word_state_mask = compact_mask_select(layer_n2, st_is('word_pivot'))
        word_n1, _ = compact_mask_select(layer_n1, st_is('word_pivot'))
        col_n2, col_state_mask = compact_mask_select(layer_n2, st_is('col_pivot'))
        col_n1, _ = compact_mask_select(layer_n1, st_is('col_pivot'))

        word_states = torch.cat([word_n2, word_n1], dim=-1)
        col_states = torch.cat([col_n2, col_n1], dim=-1)

        return word_states, word_state_mask, col_states, col_state_mask

    def get_loss(self, sel_logp, agg_logp, cond_num_logp, col_prob, ops_logp, begin_prob, end_prob,
                 select, agg, where_num, where_cols, where_ops, where_begin, where_end):
        batch_idx = torch.arange(sel_logp.size(0), device=sel_logp.device)  # (b,)

        sel_loss = - sel_logp[batch_idx, select].mean()
        agg_loss = - agg_logp[batch_idx, select, agg].mean()
        cond_num_loss = - cond_num_logp[batch_idx, where_num].mean()

        col_mask = where_cols >= 0  # where_cols contains only 0 or 1 indicating column selection, padded by -1
        # cond_col_loss = (((col_prob - where_cols) ** 2 / 2) * col_mask).sum(-1).mean()
        cond_col_loss = - ((where_cols * col_prob.clamp(min=1e-8).log()
                            + (1 - where_cols) * (1 - col_prob).clamp(min=1e-8).log())
                           * col_mask).sum(-1).mean()

        ex_batch_idx = batch_idx.unsqueeze(-1)  # (b, 1)
        ex_col_idx = torch.arange(where_ops.size(1), device=where_ops.device).unsqueeze(0)    # (1, Nc)
        ops_mask = where_ops >= 0
        ops_loss = - (ops_logp[ex_batch_idx, ex_col_idx, where_ops] * ops_mask).sum(-1).mean()

        begin_mask = where_begin >= 0
        end_mask = where_end >= 0
        begin_loss = - (begin_prob[ex_batch_idx, ex_col_idx, where_begin].clamp(min=1e-8).log()
                        * begin_mask).sum(-1).mean()
        end_loss = - (end_prob[ex_batch_idx, ex_col_idx, where_end].clamp(min=1e-8).log()
                      * end_mask).sum(-1).mean()

        losses = [sel_loss, agg_loss, cond_num_loss, cond_col_loss, ops_loss, begin_loss, end_loss]
        if self.use_metric_adaptive_losses and self.acc._count > 3200:  # >100 iterations with batch size 32
            metric_names = ('acc_select', 'acc_agg', 'acc_num', 'acc_cols', 'acc_ops', 'acc_begin', 'acc_end')
            for i, name in enumerate(metric_names):
                metric = getattr(self, name)
                if metric.get_metric() > 0.97:
                    losses[i] = 0

        loss = sum(losses)
        return loss

    def get_alignment_losses(self, word_mask, col_mask):
        if self.use_hungarian_loss == 'partial':
            attns: List[Attn] = [
                self.sel_agg.col_word_attn,
            ]
        else:
            attns: List[Attn] = [
                self.sel_col.col_word_attn,
                self.sel_agg.col_word_attn,
                self.wh_col.col_word_attn,
                self.wh_op.col_word_attn,
                self.wh_val.col_word_attn,
            ]
        losses = [
            get_hungarian_reg_loss(attn.get_latest_attn_weights(), col_mask, word_mask)
            for attn in attns
        ]
        return sum(losses)
        # attn = self.sel_agg.col_word_attn.get_latest_attn_weights()
        # return get_hungarian_reg_loss(attn, col_mask, word_mask)

    def compute_metrics(self,
                        loss, word_mask, col_mask,  # (b,)
                        sel_logp, agg_logp, cond_num_logp, col_prob, ops_logp, begin_prob, end_prob,
                        # oracle labels
                        select, agg, where_num,  # (b,)
                        where_cols, where_ops, where_begin, where_end,  # (b, Nc), padded by -1
                        ):
        self.item_count += sel_logp.size(0)
        self.mean_loss(loss)
        for n_word in word_mask.sum(dim=-1):
            self.word_num(n_word)
        for n_col in col_mask.sum(dim=-1):
            self.col_num(n_col)

        with torch.no_grad():
            pred_select = sel_logp.argmax(dim=-1)                           # (b,)
            batch = torch.arange(sel_logp.size(0), device=sel_logp.device)  # (b,)
            pred_agg = agg_logp[batch, pred_select].argmax(dim=-1)          # (b,)
            pred_num = cond_num_logp.argmax(dim=-1)                         # (b,)
            pred_cols = col_prob.clone()                                    # (b, Nc)
            pred_ops = ops_logp.argmax(dim=-1)                              # (b, Nc)
            pred_begin = begin_prob.argmax(dim=-1)                          # (b, Nc)
            pred_end = end_prob.argmax(dim=-1)                              # (b, Nc)

            acc_select = (pred_select == select)    # (b,)
            acc_agg = (pred_agg == agg)             # (b,)
            acc_num = (pred_num == where_num)       # (b,)

            is_padded_col = where_cols < 0         # (b, Nc)
            topk_cols = torch.zeros_like(is_padded_col).long()
            topk_cols[is_padded_col] = -1
            for i, (col, k, padding) in enumerate(zip(pred_cols, where_num, is_padded_col)):
                col[padding] = 0
                _, indices = torch.topk(col, k.item())
                topk_cols[i][indices] = 1
            acc_col = is_padded_col.logical_or(topk_cols == where_cols).all(dim=-1)    # (b,)

            # padded_cols can be 0 or 1, but the padded_conditions are -1 and 0 in where_cols
            is_padded_cond = where_ops < 0         # (b, Nc)
            acc_ops = is_padded_cond.logical_or(pred_ops == where_ops).all(dim=-1)         # (b,)
            acc_begin = is_padded_cond.logical_or(pred_begin == where_begin).all(dim=-1)   # (b,)
            acc_end = is_padded_cond.logical_or(pred_end == where_end).all(dim=-1)         # (b,)

            batch_acc = acc_select * acc_agg * acc_num * acc_col * acc_ops * acc_begin * acc_end

        metric_names = ('acc', 'acc_select', 'acc_agg', 'acc_num', 'acc_cols', 'acc_ops', 'acc_begin', 'acc_end')
        metric_values = (batch_acc, acc_select, acc_agg, acc_num, acc_col, acc_ops, acc_begin, acc_end)
        for name, batch_value in zip(metric_names, metric_values):
            metric = getattr(self, name)
            for item in batch_value:
                metric(item)


class WordColBiEnc(nn.Module):
    def __init__(self, word_enc: StackEncoder, col_enc: StackEncoder):
        super(WordColBiEnc, self).__init__()
        self.word_enc = word_enc
        self.col_enc = col_enc

    def forward(self, words, cols, word_mask, col_mask):
        word_h = self.word_enc.forward(words, word_mask)
        col_h = self.col_enc.forward(cols, col_mask)
        return word_h, col_h

    @classmethod
    def build(cls, inp_sz: int, hidden: int, num_layers: int, dropout: float):
        word_enc = EncoderStackMixin.get_stacked_rnn_encoder('bilstm', inp_sz, hidden // 2, num_layers, dropout)
        col_enc = EncoderStackMixin.get_stacked_rnn_encoder('bilstm', inp_sz, hidden // 2, num_layers, dropout)
        return cls(word_enc, col_enc)


class SelectColumn(nn.Module):
    def __init__(self, enc: WordColBiEnc, col_word_attn: Attn, hidden: int):
        super(SelectColumn, self).__init__()
        self.enc = enc
        self.col_word_attn = col_word_attn
        self.map_ctx = nn.Linear(hidden, hidden)
        self.map_col = nn.Linear(hidden, hidden)
        self.map_cat = nn.Sequential(nn.Tanh(), nn.Linear(hidden * 2, 1))

    def forward(self, words, cols, word_mask, col_mask):
        words, cols = self.enc(words, cols, word_mask, col_mask)
        mapped_ctx = self.map_ctx(self.col_word_attn(cols, words, word_mask))               # (b, Nc, h)
        mapped_col = self.map_col(cols)                                                     # (b, Nc, h)
        col_score = self.map_cat(torch.cat([mapped_col, mapped_ctx], dim=-1)).squeeze(-1)   # (b, Nc)
        return masked_log_softmax(col_score, col_mask, dim=-1)  # (b, Nc), log-prob of column selection


class SelectAgg(nn.Module):
    def __init__(self, enc: WordColBiEnc, col_word_attn: Attn, hidden: int, num_aggs):
        super(SelectAgg, self).__init__()
        self.enc = enc
        self.col_word_attn = col_word_attn
        self.agg_scorer = nn.Sequential(nn.Linear(hidden, hidden),
                                        nn.Tanh(),
                                        nn.Linear(hidden, num_aggs),
                                        nn.LogSoftmax(dim=-1))

    def forward(self, words, cols, word_mask, col_mask):
        words, cols = self.enc(words, cols, word_mask, col_mask)    # (b, Nw, h), (b, Nc, h)
        mapped_ctx = self.col_word_attn(cols, words, word_mask)     # (b, Nc, h)
        return self.agg_scorer(mapped_ctx)                          # (b, Nc, Naggs), log-prob of different aggs


class WhereNumber(nn.Module):
    def __init__(self, enc: WordColBiEnc, hidden: int, num_layers: int, max_where_number: int, ):
        super(WhereNumber, self).__init__()
        self.enc = enc
        self.self_attn = nn.Linear(hidden, 1)
        self.num_layers = num_layers
        self.hidden_size = hidden
        self.map_h = nn.Linear(hidden, hidden * num_layers)
        self.map_c = nn.Linear(hidden, hidden * num_layers)
        self.lstm_attn = nn.LSTM(hidden, hidden // 2, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.map_rnn_out = nn.Linear(hidden, 1)
        self.scorer = nn.Sequential(nn.Linear(hidden, hidden),
                                    nn.Tanh(),
                                    nn.Linear(hidden, 1 + max_where_number),
                                    nn.LogSoftmax(dim=-1))

    def forward(self, words, cols, word_mask, col_mask):
        words, cols = self.enc(words, cols, word_mask, col_mask)    # (b, Nw, h), (b, Nc, h)
        self_attn_logits = self.self_attn(cols).squeeze(-1)         # (b, Nc)
        col_ctx = (masked_softmax(self_attn_logits, col_mask).unsqueeze(-1) * cols).sum(dim=1)  # (b, h)
        bsz, nhid = col_ctx.size()
        # (L, b, h) <- (b, L * direction, h // direction)
        mapped_h = self.map_h(col_ctx).reshape(bsz, self.num_layers * 2, nhid // 2).transpose(0, 1).contiguous()
        mapped_c = self.map_c(col_ctx).reshape(bsz, self.num_layers * 2, nhid // 2).transpose(0, 1).contiguous()

        hids, _ = self.lstm_attn(words, (mapped_h, mapped_c))  # (b, Nw, h)
        mapped_hids = torch.softmax(self.map_rnn_out(hids), dim=-2)  # (b, Nw, 1)
        word_ctx = (words * mapped_hids).sum(dim=-2)   # (b, dw)
        return self.scorer(word_ctx)   # (b, 1 + max_where), in logarithm


class WhereColumn(nn.Module):
    def __init__(self, enc: WordColBiEnc, col_word_attn: Attn, hidden: int):
        super().__init__()
        self.enc = enc
        self.col_word_attn = col_word_attn
        self.map_ctx = nn.Linear(hidden, hidden)
        self.map_col = nn.Linear(hidden, hidden)
        self.scorer = nn.Sequential(nn.Tanh(), nn.Linear(hidden * 2, 1), nn.Sigmoid())

    def forward(self, words, cols, word_mask, col_mask):
        words, cols = self.enc(words, cols, word_mask, col_mask)
        mapped_ctx = self.map_ctx(self.col_word_attn(cols, words, word_mask))   # (b, Nc, h)
        mapped_col = self.map_col(cols)                                         # (b, Nc, h)
        col_repr = torch.cat([mapped_col, mapped_ctx], dim=-1)                  # (b, Nc, h*2)
        return self.scorer(col_repr).squeeze(-1)    # (b, Nc) <-- (b, Nc, 1), probabilities by sigmoid


class WhereOps(nn.Module):
    def __init__(self, enc: WordColBiEnc, col_word_attn: Attn, hidden: int, num_ops: int,):
        super().__init__()
        self.enc = enc
        # self.col_word_attn = Attn(AllenLogits(BilinearMatrixAttention(hidden, hidden)))
        self.col_word_attn = col_word_attn
        self.map_ctx = nn.Linear(hidden, hidden)
        self.map_col = nn.Linear(hidden, hidden)
        self.map_cat = nn.Linear(hidden * 2, hidden)
        self.scorer = nn.Sequential(nn.Tanh(), nn.Linear(hidden, num_ops), nn.LogSoftmax(dim=-1))

    def forward(self, words, cols, word_mask, col_mask):
        words, cols = self.enc(words, cols, word_mask, col_mask)
        mapped_ctx = self.map_ctx(self.col_word_attn(cols, words, word_mask))   # (b, Nc, h)
        mapped_col = self.map_col(cols)  # (b, Nc, h)
        mapped_cat = self.map_cat(torch.cat([mapped_col, mapped_ctx], dim=-1))  # (b, Nc, h)
        return self.scorer(mapped_cat)  # (b, Nc, num_ops)


class WhereValue(nn.Module):
    def __init__(self, enc: WordColBiEnc, col_word_attn: Attn, hidden: int, num_ops: int,
                 begin_scorer: Attn,
                 end_scorer: Attn,
                 ):
        super().__init__()
        self.enc = enc
        self.col_word_attn = col_word_attn
        self.map_ctx = nn.Linear(hidden, hidden)
        self.map_col = nn.Linear(hidden, hidden)
        self.map_ops = nn.Linear(num_ops, hidden)
        self.begin_scorer = begin_scorer
        self.end_scorer = end_scorer

    def forward(self, words, cols, word_mask, col_mask, op_soft):
        words, cols = self.enc(words, cols, word_mask, col_mask)                # (b, Nw, h), (b, Nc, h)
        mapped_ctx = self.map_ctx(self.col_word_attn(cols, words, word_mask))   # (b, Nc, h)
        mapped_col = self.map_col(cols)     # (b, Nc, h)
        mapped_ops = self.map_ops(op_soft)  # (b, Nc, h) <-- (b, Nc, num_ops)

        col_repr = torch.cat([mapped_col, mapped_ctx, mapped_ops], dim=-1)  # (b, Nc, h * 3)

        self.begin_scorer(col_repr, words, word_mask)
        begin_score = self.begin_scorer.get_latest_attn_weights()   # (b, Nc, Nw)
        self.end_scorer(col_repr, words, word_mask)
        end_score = self.end_scorer.get_latest_attn_weights()       # (b, Nc, Nw)

        return begin_score, end_score
