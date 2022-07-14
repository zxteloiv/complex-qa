import torch
from allennlp.training.metrics import Average
from allennlp.nn.util import masked_log_softmax
from torch import nn

from models.interfaces.attention import Attention as IAttn
from models.interfaces.encoder import Encoder
from models.interfaces.unified_rnn import RNNStack
from models.modules.variational_dropout import VariationalDropout
from models.transformer.multi_head_attention import MultiHeadAttention
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

                 # optional modules
                 aux_col: IAttn = None,

                 # configs
                 src_type_keys: tuple = ('pad', 'special', 'word', 'word_pivot', 'column', 'col_pivot'),
                 tgt_type_keys: tuple = ('pad', 'keyword', 'column', 'literal_string', 'literal_number'),
                 decoder_init_strategy: str = "zero_all"
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

        self._strategy = decoder_init_strategy

        self.stype_map = dict(zip(src_type_keys, range(len(src_type_keys))))
        self.ttype_map = dict(zip(tgt_type_keys, range(len(tgt_type_keys))))
        self.acc = Average()
        self.word_num = Average()
        self.col_num = Average()
        self.tgt_len = Average()
        self.mean_loss = Average()
        self.item_count = 0
        self.match_stats = None

    def get_metrics(self, reset=False):
        metric = {"ACC(EM)": round(self.acc.get_metric(reset), 6),
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

    def _reset_variational_dropouts(self):
        for m in self.modules():
            if isinstance(m, VariationalDropout):
                m.reset()

    def forward(self,
                # encoder inputs
                src_ids, src_types: torch.Tensor, src_plm_type_ids,
                # alignments between each word/sql, word/col, and sql/col pair
                align_ws_word, align_ws_sql, align_wc_word, align_wc_col, align_sc_sql, align_sc_col,
                # decoder supervisions,
                tgt_type, tgt_keyword, tgt_col_id, tgt_col_type, tgt_literal_begin, tgt_literal_end,
                # extension for future uses
                **kwargs
                ):
        self._reset_variational_dropouts()
        word_states, word_mask, col_states, col_mask = self.encode(src_ids, src_types, src_plm_type_ids)

        hx = self.init_decoder(word_states, word_mask)

        mem = SeqCollector()
        for i in range(tgt_type.size()[1] - 1):
            inp = self.get_teacher_forcing_input(
                i, tgt_type, word_states, col_states, tgt_keyword, tgt_col_id,
                tgt_literal_begin, tgt_literal_end
            )

            hx, out = self.decoder(inp, hx)
            word_ctx, col_ctx = self.get_enc_attn(out, word_states, word_mask, col_states, col_mask)

            hvec = torch.cat([out, word_ctx, col_ctx], dim=-1)   # (b, hidden * 3)
            self.step_predictions(hvec, word_states, word_mask, col_states, col_mask, mem)

        losses, matches, match_stat = self.get_decoder_loss(
            mem, tgt_type[:, 1:], tgt_keyword[:, 1:], tgt_col_id[:, 1:], tgt_col_type[:, 1:],
            tgt_literal_begin[:, 1:], tgt_literal_end[:, 1:], align_ws_word, align_ws_sql,
            col_type_mask=kwargs.get('col_type_mask'),
        )
        loss_enc_attn = self.get_encoder_loss(align_wc_word, align_wc_col)
        losses.append(loss_enc_attn)
        if self.aux_word_col is not None:
            aux_loss = self.get_aux_loss(word_states, col_states, col_mask, align_wc_word, align_wc_col)
            losses.append(0.2 * aux_loss)
        final_loss = sum(losses)

        self.compute_metrics(word_mask, col_mask, matches, match_stat, tgt_type[:, 1:], final_loss)

        output = {"src_id": src_ids, "src_type": src_types, "tgt_type": tgt_type,
                  "tgt_keyword": tgt_keyword, "tgt_col_id": tgt_col_id, "tgt_col_type": tgt_col_type,
                  "tgt_literal_begin": tgt_literal_begin, "tgt_literal_end": tgt_literal_end,
                  "loss": final_loss,
                  }
        return output

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

    def get_decoder_loss(self, mem: SeqCollector, gold_type: torch.LongTensor,
                         gold_keyword, gold_col_id, gold_col_type,
                         gold_span_begin, gold_span_end,
                         ws_word, ws_sql,
                         col_type_mask: torch.LongTensor = None,
                         ):
        # memory have the following keys as in the self.step_predictions method
        # mem(ttype=ttype_logit, keyword=keyword_logit, col_type=col_type_logit, word_attn=word_attn_weights,
        #     col_copy=col_copy_logit, span_begin=span_begin_logit, span_end=span_end_logit)

        losses = []
        matches = []
        nbatch, dev = gold_type.size(0), gold_type.device
        match_stat = []

        def _add_loss(logprob, target, mask: torch.BoolTensor):
            losses.append(-masked_reducing_gather(logprob, target, mask.float(), 'batch'))
            correct = logprob.argmax(dim=-1) == target
            matches.append(torch.bitwise_or(~mask, correct).all(dim=-1))
            nominator = torch.bitwise_and(mask, correct).sum().item()
            denominator = mask.sum().item()
            match_stat.append([nominator, denominator])

        def is_type(k: str):
            return gold_type == self.ttype_map[k]

        # logits based prediction loss
        # *_logit: (b, tgt_len, #V), the unnormalized logits over #V
        # *_mask: (b, tgt_len)
        _add_loss(logprob(mem.get_stacked_tensor('ttype')), gold_type, ~is_type('pad'))
        _add_loss(logprob(mem.get_stacked_tensor('keyword')), gold_keyword, is_type('keyword'))

        # col_type_mask: (b, col_num, #V)
        batch = torch.arange(nbatch, device=dev).unsqueeze(-1)  # (b, 1)
        if self.training:
            masks = None
        else:
            masks = col_type_mask[batch, gold_col_id]   # (b, tgt_len, #V)
        _add_loss(logprob(mem.get_stacked_tensor('col_type'), masks), gold_col_type, is_type('column'))

        # copy-based prediction losses, with gold selection labels
        # (b, tgt_len, src_len)
        _add_loss(log(mem.get_stacked_tensor('col_copy')), gold_col_id, is_type('column'))

        span_begin_log = log(mem.get_stacked_tensor('span_begin'))
        _add_loss(span_begin_log, gold_span_begin, is_type('literal_number'))
        _add_loss(span_begin_log, gold_span_begin, is_type('literal_string'))
        span_end_prob = mem.get_stacked_tensor('span_end')  # (b, tgt_len, src_len)
        if self.training:
            span_end_logp = log(span_end_prob)
        else:
            span_end_mask = torch.ones_like(span_end_prob)
            for i in range(nbatch):
                for j, step_begin in enumerate(gold_span_begin[i]):
                    span_end_mask[i, j, :step_begin] = 0
            span_end_norm = (span_end_prob * span_end_mask).sum(-1, keepdims=True).clamp(min=1e-20)
            span_end_logp = (span_end_prob / span_end_norm).clamp(min=1e-20).log()
        _add_loss(span_end_logp, gold_span_end, is_type('literal_string'))

        # additional attention supervision losses, with gold alignments
        # (b, tgt_len, src_len)
        word_attn = mem.get_stacked_tensor('word_attn')
        # attention as supervision but not prediction targets
        dup_sql_num = (ws_sql.unsqueeze(-1) == ws_sql.unsqueeze(-2)).sum(-1)
        word_prob = word_attn[batch, ws_sql, ws_word]
        losses.append(((word_prob - dup_sql_num.reciprocal()) ** 2 / 2).sum(-1).mean()) # sum alignments, mean batch
        return losses, matches, match_stat

    def encode(self, src_ids, src_types: torch.Tensor, plm_type_ids):
        state_mask: torch.Tensor = (src_types != self.stype_map['pad'])
        enc_out = self.pretrained_model(input_ids=src_ids,
                                        token_type_ids=plm_type_ids,
                                        attention_mask=state_mask.float())
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

    def get_encoder_loss(self, align_word, align_col):
        # attn: (b, n, 1, attend_length)
        batch = torch.arange(align_word.size()[0], device=align_word.device).unsqueeze(-1)

        col_attn = self.word_col_attn.get_latest_attn_weights()
        word_attn = self.col_word_attn.get_latest_attn_weights()

        # align_word/col: (b, num_alignments)
        # (batch, num_alignments)
        col_probs = col_attn[batch, align_word, align_col]   # each word to col distribution
        word_probs = word_attn[batch, align_col, align_word] # each col to words distribution

        dup_word_num = (align_word.unsqueeze(-1) == align_word.unsqueeze(-2)).sum(-1)
        dup_col_num = (align_col.unsqueeze(-1) == align_col.unsqueeze(-2)).sum(-1)

        col_loss = (col_probs - dup_word_num.reciprocal()) ** 2 / 2
        word_loss = (word_probs - dup_col_num.reciprocal()) ** 2 / 2
        return (col_loss + word_loss).sum(-1).mean()

    def get_aux_loss(self, word_states, col_states, col_mask, align_word, align_col):
        # NLL loss for the the attention module
        _ = self.aux_word_col.forward(word_states, col_states, col_mask)
        col_attn = self.aux_word_col.get_latest_attn_weights()  # (b, word_len, col_len)

        batch = torch.arange(align_word.size()[0], device=align_word.device).unsqueeze(-1)
        col_probs = col_attn[batch, align_word, align_col]   # (batch, num_alignments)
        col_logprobs = col_probs.clamp(min=1e-20).log()     # (batch, num_alignments)
        aux_loss = - col_logprobs.sum(-1).mean()
        return aux_loss

    def init_decoder(self, states, state_mask):
        agg_src = aggregate_layered_state([states], state_mask, self._strategy)
        dec_states = assign_stacked_states(agg_src, self.decoder.get_layer_num(), self._strategy)
        # init the internal hiddens
        hx, _ = self.decoder.init_hidden_states(dec_states)
        return hx

    def get_teacher_forcing_input(self, step: int, tgt_type: torch.Tensor, word_states, col_states,
                                  keywords, col_id, span_begin, span_end):
        batch = tgt_type.size()[0]
        step_input_type = tgt_type[:, step].unsqueeze(-1)   # (batch, 1)

        # *_inp, span_*: (batch, hid)
        emb_inp = self.sql_keyword_embed(keywords[:, step])
        col_inp = self.col2input(col_states[range(batch), col_id[:, step]])
        span_begin_state = word_states[range(batch), span_begin[:, step]]
        span_end_state = word_states[range(batch), span_end[:, step]]
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

        mem(ttype=ttype_logit, keyword=keyword_logit, col_type=col_type_logit, word_attn=word_attn_prob,
            col_copy=col_copy_prob, span_begin=span_begin_prob, span_end=span_end_prob)


def logprob(logits, mask=None):
    if mask is None:
        return torch.log_softmax(logits, dim=-1)
    return masked_log_softmax(logits, mask, dim=-1)


def log(prob):
    return prob.clamp(min=1e-20).log()

