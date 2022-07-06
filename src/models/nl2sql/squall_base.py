import torch
from allennlp.training.metrics import Average
from torch import nn

from models.interfaces.attention import Attention as IAttn
from models.interfaces.unified_rnn import RNNStack
from models.modules.variational_dropout import VariationalDropout
from models.transformer.multi_head_attention import MultiHeadSelfAttention
from utils.nn import aggregate_layered_state, assign_stacked_states, masked_reducing_gather
from utils.seq_collector import SeqCollector


class SquallBaseParser(nn.Module):
    def __init__(self,
                 # modules
                 plm_model,
                 hidden_sz: int,
                 plm2hidden: nn.Module,

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

                 # configs
                 src_type_keys: tuple = ('pad', 'special', 'word', 'word_pivot', 'column', 'col_pivot'),
                 tgt_type_keys: tuple = ('pad', 'keyword', 'column', 'literal_string', 'literal_number'),
                 decoder_init_strategy: str = "zero_all"
                 ):
        super().__init__()
        self.pretrained_model = plm_model
        self.plm2hidden = plm2hidden

        self.aux_col = MultiHeadSelfAttention(1, hidden_sz, hidden_sz, hidden_sz, 0)
        self.aux_word = MultiHeadSelfAttention(1, hidden_sz, hidden_sz, hidden_sz, 0)

        self.sql_word_attn = sql_word_attn
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
        self.item_count = 0

    def get_metrics(self, reset=False):
        metric = {"ACC(EM)": self.acc.get_metric(reset),
                  "WORDS": self.word_num.get_metric(reset),
                  "COLUMNS": self.col_num.get_metric(reset),
                  "TLEN": self.tgt_len.get_metric(reset),
                  "COUNT": self.item_count,
                  }
        if reset:
            self.item_count = 0
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
        enc_states, state_mask = self.encode(src_ids, src_types, src_plm_type_ids)
        hx = self.init_decoder(enc_states, state_mask)

        col_pivot = (src_types == self.stype_map['col_pivot']).long()
        word_pivot = (src_types == self.stype_map['word_pivot']).long()

        loss_aux = self.aux_pred(enc_states, word_pivot, col_pivot, align_wc_word, align_wc_col)

        mem = SeqCollector()
        for i in range(tgt_type.size()[1] - 1):
            inp = self.get_teacher_forcing_input(0, tgt_type, enc_states, tgt_keyword, tgt_col_id,
                                                 tgt_literal_begin, tgt_literal_end)
            hx, out = self.decoder(inp, hx)
            word_ctx, col_ctx = self.get_enc_attn(out, word_pivot, col_pivot, enc_states)

            hvec = torch.cat([out, word_ctx, col_ctx], dim=-1)   # (b, hidden * 3)
            self.step_predictions(hvec, enc_states, word_pivot, col_pivot, mem)

        losses, matches = self.get_training_loss(mem, tgt_type[:, 1:], tgt_keyword[:, 1:],
                                                 tgt_col_id[:, 1:], tgt_col_type[:, 1:],
                                                 tgt_literal_begin[:, 1:], tgt_literal_end[:, 1:],
                                                 align_ws_word, align_ws_sql)
        losses.append(0.2 * loss_aux)
        self.compute_metrics(word_pivot, col_pivot, matches, tgt_type[:, 1:])

        final_loss = sum(losses)

        output = {"src_id": src_ids, "src_type": src_types, "tgt_type": tgt_type,
                  "tgt_keyword": tgt_keyword, "tgt_col_id": tgt_col_id, "tgt_col_type": tgt_col_type,
                  "tgt_literal_begin": tgt_literal_begin, "tgt_literal_end": tgt_literal_end,
                  "loss": final_loss,
                  }
        return output

    def compute_metrics(self, word_pivot, col_pivot, matches, tgt_type):
        for n in word_pivot.sum(-1):
            self.word_num(n)
        for n in col_pivot.sum(-1):
            self.col_num(n)
        for m in torch.stack(matches, dim=1).all(dim=1).long():
            self.acc(m)
        for n in (tgt_type == self.ttype_map['pad']).sum(-1):
            self.tgt_len(n)
        self.item_count += word_pivot.size()[0]

    def get_training_loss(self, mem: SeqCollector, gold_type: torch.LongTensor,
                          gold_keyword, gold_col_id, gold_col_type,
                          gold_span_begin, gold_span_end,
                          ws_word, ws_sql,
                          ):
        # memory have the following keys as in the self.step_predictions method
        # mem(ttype=ttype_logit, keyword=keyword_logit, col_type=col_type_logit, word_attn=word_attn_weights,
        #     col_copy=col_copy_logit, span_begin=span_begin_logit, span_end=span_end_logit)

        losses = []
        matches = []

        def logprob(logits):
            return torch.log_softmax(logits, dim=-1)

        def log(prob):
            return (prob + 1e-13).log()

        def _append_loss(logprob, target, mask: torch.BoolTensor):
            losses.append(-masked_reducing_gather(logprob, target, mask.float(), 'batch'))
            matches.append(torch.bitwise_or(~mask, logprob.argmax(dim=-1) == target).all(dim=-1))

        def is_type(k: str):
            return gold_type == self.ttype_map[k]

        # logits based losses
        # *_logit: (b, tgt_len, #V), the unnormalized logits over #V
        # *_mask: (b, tgt_len)
        _append_loss(logprob(mem.get_stacked_tensor('ttype')), gold_type, ~is_type('pad'))
        _append_loss(logprob(mem.get_stacked_tensor('keyword')), gold_keyword, is_type('keyword'))
        _append_loss(logprob(mem.get_stacked_tensor('col_type')), gold_col_type, is_type('column'))

        # attention-based losses with gold selection labels
        # (b, tgt_len, src_len)
        _append_loss(log(mem.get_stacked_tensor('col_copy')), gold_col_id, is_type('column'))

        span_begin_log = log(mem.get_stacked_tensor('span_begin'))
        _append_loss(span_begin_log, gold_span_begin, is_type('literal_number'))
        _append_loss(span_begin_log, gold_span_begin, is_type('literal_string'))
        _append_loss(log(mem.get_stacked_tensor('span_end')), gold_span_end, is_type('literal_string'))

        # attention-based losses with gold alignments
        # (b, tgt_len, src_len)
        word_attn_logp = log(mem.get_stacked_tensor('word_attn'))
        batch = torch.arange(gold_type.size()[0], device=gold_type.device).unsqueeze(-1)
        # attention as supervision but not prediction targets
        align_mask = ws_word > 0    # (b, num_alignments)
        losses.append(- (word_attn_logp[batch, ws_sql, ws_word] * align_mask).sum(-1).mean())

        return losses, matches

    def encode(self, src_ids, src_types: torch.Tensor, plm_type_ids):
        state_mask: torch.Tensor = (src_types != self.stype_map['pad'])
        enc_out = self.pretrained_model(input_ids=src_ids,
                                        token_type_ids=plm_type_ids,
                                        attention_mask=state_mask.float())
        enc_states = enc_out.last_hidden_state
        states = self.plm2hidden(enc_states)
        return states, state_mask

    def aux_pred(self, states: torch.Tensor, word_pivot, col_pivot, align_word, align_col):
        # states: (b, n, hid)
        # preds: (b, n, 1, attend_length)
        _, col_preds = self.aux_col.forward(states, col_pivot)
        _, word_preds = self.aux_word.forward(states, word_pivot)

        batch = torch.arange(states.size()[0], device=states.device).unsqueeze(-1)

        # align_word/col: (b, num_alignments)
        # (batch, num_alignments)
        col_probs = col_preds[batch, align_word, 0, align_col]
        word_probs = word_preds[batch, align_col, 0, align_word]

        align_mask = (align_word > 0)

        col_loss = - ((col_probs + 1e-13).log() * align_mask).sum(-1).mean()
        word_loss = - ((word_probs + 1e-13).log() * align_mask).sum(-1).mean()
        return col_loss + word_loss

    def init_decoder(self, states, state_mask):
        agg_src = aggregate_layered_state([states], state_mask, self._strategy)
        dec_states = assign_stacked_states(agg_src, self.decoder.get_layer_num(), self._strategy)
        # init the internal hiddens
        hx, _ = self.decoder.init_hidden_states(dec_states)
        return hx

    def get_teacher_forcing_input(self, step: int, tgt_type: torch.Tensor, src_states,
                                  keywords, col_id, span_begin, span_end):
        batch = tgt_type.size()[0]
        step_input_type = tgt_type[:, step].unsqueeze(-1)   # (batch, 1)

        # *_inp, span_*: (batch, hid)
        emb_inp = self.sql_keyword_embed(keywords[:, step])
        col_inp = self.col2input(src_states[range(batch), col_id[:, step]])
        span_begin_state = src_states[range(batch), span_begin[:, step]]
        span_end_state = src_states[range(batch), span_end[:, step]]
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

    def get_enc_attn(self, step_out, word_pivot, col_pivot, src_states):
        # step_out: (b, hid)
        # *_pivot: (b, n)
        word_context = self.sql_word_attn.forward(step_out, src_states, word_pivot)
        col_context = self.sql_col_attn.forward(step_out, src_states, col_pivot)
        # attn_weights for word_context will be added in self.step_predictions for training supervision
        # and thus will not be returned here.
        return word_context, col_context

    def step_predictions(self, hvec, enc_states, word_pivot, col_pivot, mem):
        # classification-based predictions
        ttype_logit = self.sql_type(hvec)  # (b, #types)
        keyword_logit = self.sql_keyword(hvec)  # (b, #keywords)
        col_type_logit = self.sql_col_type(hvec)  # (b, #coltypes)

        # attention-based predictions
        self.sql_col_copy.forward(hvec, enc_states, col_pivot)
        col_copy_prob = self.sql_col_copy.get_latest_attn_weights()  # (b, n)
        self.sql_span_begin.forward(hvec, enc_states, word_pivot)
        span_begin_prob = self.sql_span_begin.get_latest_attn_weights()  # (b, n)
        self.sql_span_end.forward(hvec, enc_states, word_pivot)
        span_end_prob = self.sql_span_end.get_latest_attn_weights()  # (b, n)
        # for encoder attention training supervision
        word_attn_prob = self.sql_word_attn.get_latest_attn_weights()    # (b, n)

        mem(ttype=ttype_logit, keyword=keyword_logit, col_type=col_type_logit, word_attn=word_attn_prob,
            col_copy=col_copy_prob, span_begin=span_begin_prob, span_end=span_end_prob)
