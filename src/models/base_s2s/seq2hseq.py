from typing import Union, Optional, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn
import torch.nn.functional
from trialbot.data.ns_vocabulary import NSVocabulary
from ..interfaces.attention import Attention as IAttn, VectorContextComposer as AttnComposer
from ..modules.variational_dropout import VariationalDropout
from .base_seq2seq import BaseSeq2Seq
from .stacked_rnn_cell import StackedRNNCell
from .stacked_encoder import StackedEncoder
from utils.nn import prepare_input_mask, seq_cross_ent, seq_likelihood, masked_reducing_gather
from utils.nn import init_stacked_dec_state_from_enc, init_state_for_stacked_rnn
from utils.nn import aggregate_layered_state, assign_stacked_states
from utils.seq_collector import SeqCollector
from utils.text_tool import make_human_readable_text
from allennlp.training.metrics import BLEU, Perplexity, Average


class Seq2HierSeq(torch.nn.Module):
    def __init__(self,
                 # modules
                 vocab: NSVocabulary,
                 source_embedding: torch.nn.Embedding,
                 target_embedding: torch.nn.Embedding,

                 encoder: StackedEncoder,

                 word_decoder: StackedRNNCell,
                 char_decoder: StackedRNNCell,

                 word_projector: torch.nn.Module,
                 seq_len_predictor: torch.nn.Module,

                 enc_attn_word: IAttn,
                 enc_attn_char: IAttn,

                 word_dec_inp_attn_comp: AttnComposer,
                 char_dec_inp_attn_comp: AttnComposer,
                 proj_inp_attn_comp: AttnComposer,

                 # model configuration
                 source_namespace: str = "source_tokens",
                 target_namespace: str = "target_tokens",
                 start_symbol: str = '<GO>',
                 eos_symbol: str = '<EOS>',
                 padding_index: int = 0,

                 max_word_step: int = 50,
                 max_char_step: int = 10,

                 word_decoder_init_strategy: str = "forward_all",

                 # training_configuration
                 scheduled_sampling_ratio: float = 0.,
                 enc_dropout: float = 0,
                 dec_dropout: float = .1,
                 ):
        super().__init__()
        self.vocab = vocab

        self.word_enc_attn = enc_attn_word
        self.char_enc_attn = enc_attn_char

        self.word_dec_inp_attn_comp = word_dec_inp_attn_comp
        self.char_dec_inp_attn_comp = char_dec_inp_attn_comp

        self.proj_inp_attn_comp = proj_inp_attn_comp

        self.scheduled_sampling_ratio = scheduled_sampling_ratio
        self.encoder = encoder

        self.word_decoder: StackedRNNCell = word_decoder
        self.char_decoder: StackedRNNCell = char_decoder

        self.src_emb_layer = source_embedding
        self.tgt_emb_layer = target_embedding

        self.src_ns = source_namespace
        self.tgt_ns = target_namespace
        self.start_id = vocab.get_token_index(start_symbol, target_namespace)
        self.eos_id = vocab.get_token_index(eos_symbol, target_namespace)

        self.max_word_step = max_word_step
        self.max_char_step = max_char_step

        self.word_proj_layer = word_projector
        self.word_stop_predictor = seq_len_predictor

        self.src_emb_dropout = VariationalDropout(enc_dropout, on_the_fly=True)
        self.tgt_emb_dropout = VariationalDropout(dec_dropout, on_the_fly=False)
        self.proj_inp_dropout = VariationalDropout(dec_dropout, on_the_fly=False)

        self.padding_index = padding_index
        self.strategy = word_decoder_init_strategy

        self.bleu = BLEU(exclude_indices={padding_index, self.start_id, self.eos_id})
        self.ppl = Perplexity()
        self.err_rate = Average()
        self.src_len = Average()
        self.tgt_len = Average()
        self.item_count = 0

    def get_metric(self, reset=False):
        metric = {"PPL": self.ppl.get_metric(reset),
                  "ERR": self.err_rate.get_metric(reset),
                  "COUNT": self.item_count,
                  "SLEN": self.src_len.get_metric(reset),
                  "TLEN": self.tgt_len.get_metric(reset),
                  }
        metric.update(self.bleu.get_metric(reset))
        if reset:
            self.item_count = 0

        return metric

    # for possible misspelling error
    get_metrics = get_metric

    def forward(self,
                source_tokens: torch.LongTensor,
                target_tokens: torch.LongTensor = None,
                *args, **kwargs
                ) -> Dict[str, torch.Tensor]:
        # src: (batch, source_length), containing the input word IDs
        # tgt: (batch, target_length, max_char_length), containing the output IDs
        src, src_mask = prepare_input_mask(source_tokens, self.padding_index)
        tgt, tgt_mask = prepare_input_mask(target_tokens, self.padding_index)

        output = {"source": src}
        self._reset_variational_dropouts()

        state, layer_states = self._encode(src, src_mask)
        word_attn_fn, char_attn_fn = self._get_enc_attn_fn(state, src_mask)
        hx = self._init_word_decoder(layer_states, src_mask)

        assert not self.training or tgt is not None
        # preds: (batch, num_words, num_chars)
        # logits: (batch, num_words, num_chars, vocab_size)
        # stop_probs: (B, Lw + 1), probs between (0, 1)
        preds, logits, stop_probs = self._forward_loop(src_mask, word_attn_fn, char_attn_fn, hx, tgt)
        if self.training:
            loss1, loss2 = self._get_loss(logits, stop_probs, tgt, tgt_mask)
            output['loss'] = loss1 + loss2
            output['loss1'] = loss1.detach()
            output['loss2'] = loss2.detach()

        output.update(predictions=preds, logits=logits, target=tgt)
        if tgt is not None:
            total_err = self._compute_metrics(src_mask, preds, logits, tgt, tgt_mask)
            output.update(errno=total_err.tolist())

        return output

    def _get_loss(self, logits: torch.FloatTensor, stop_probs: torch.Tensor,
                  tgt: torch.Tensor, tgt_mask: torch.Tensor):
        """
        :param logits: (B, Lw, Lc - 1, V)
        :param stop_probs: (B, Lw + 1)
        :param tgt: (B, Lw, Lc)
        :param tgt_mask: (B, Lw, Lc)
        :return:
        """
        loss_pred = seq_cross_ent(logits, tgt[:, :, 1:].contiguous(), tgt_mask[:, :, 1:].contiguous())

        # word_mask: (B, Lw), most valid words except the paddings are non-stopping 1
        word_mask = (tgt_mask.sum(-1) > 0).long()
        # the generation at the next location (Lw + 1) must be stopped (0)
        aux_mask = tgt_mask.new_zeros((tgt_mask.size()[0], 1))
        # stop_tgt: (B, Lw + 1)
        stop_tgt = torch.cat([word_mask, aux_mask], dim=-1)
        loss_stop = torch.nn.functional.binary_cross_entropy(stop_probs, stop_tgt.float())

        return loss_pred, loss_stop

    def _reset_variational_dropouts(self):
        self.word_decoder.reset()
        self.char_decoder.reset()
        self.tgt_emb_dropout.reset()
        self.proj_inp_dropout.reset()

    def _encode(self, source: torch.LongTensor, source_mask: torch.LongTensor):
        # source: (batch, max_input_length), source sequence token ids
        # source_mask: (batch, max_input_length), source sequence padding mask
        # source_embedding: (batch, max_input_length, embedding_sz)
        source_embedding = self.src_emb_layer(source)
        source_embedding = self.src_emb_dropout(source_embedding)
        source_hidden, layered_hidden = self.encoder(source_embedding, source_mask)
        return source_hidden, layered_hidden

    def _get_enc_attn_fn(self, source_state, source_mask):
        word_attn_fn = char_attn_fn = lambda out: None
        if self.word_enc_attn is not None:
            word_attn_fn = lambda out: self.word_enc_attn(out, source_state, source_mask)
        if self.char_enc_attn is not None:
            char_attn_fn = lambda out: self.char_enc_attn(out, source_state, source_mask)
        return word_attn_fn, char_attn_fn

    def _init_word_decoder(self, layer_states, source_mask) -> Any:
        hx, _ = self.word_decoder.init_hidden_states(init_stacked_dec_state_from_enc(
            layer_states,
            source_mask,
            self.strategy,
            self.encoder.is_bidirectional(),
            self.word_decoder.get_layer_num(),
        ))
        return hx

    def _init_char_decoder(self, word_cell_out):
        hx, _ = self.char_decoder.init_hidden_states(init_state_for_stacked_rnn(
            [word_cell_out],
            self.char_decoder.get_layer_num(),
            "lowest"
        ))
        return hx

    def _forward_loop(self, source_mask: torch.Tensor,
                      word_attn_fn, # tensor -> tensor
                      char_attn_fn, # tensor -> tensor
                      word_hx: Any,
                      target: Optional[torch.LongTensor],   # (B, Lw, Lc)
                      ) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor]:

        batch_sz = source_mask.size()[0]
        start_tok = source_mask.new_full((batch_sz,), fill_value=self.start_id)

        num_word_step = self._get_num_steps(target, is_for_word=True)
        num_char_step = self._get_num_steps(target, is_for_word=False)

        # the word input is zero-initialized with the same size as a normal output
        # to keep consistent with the future inputs.
        # otherwise, say, initializing it with an embedding will require the embedding layer
        # sharing the same dimension with the decoder output, which we wanna avoid.
        word_cell_out = torch.zeros_like(self.word_decoder.get_output_state(word_hx))
        mem = SeqCollector()
        for word_step in range(num_word_step + 1):
            word_cell_inp = self._get_cell_input(word_cell_out, word_hx, word_attn_fn, is_for_word=True)
            word_hx, word_cell_out = self.word_decoder(word_cell_inp, word_hx)
            step_stop = self.word_stop_predictor(word_cell_out)
            mem(word_stop=step_stop)
            if word_step == num_word_step:
                break   # skip the following generations

            char_hx = self._init_char_decoder(word_cell_out)
            self._forward_char_loop(start_tok, target, word_step, char_hx, char_attn_fn, mem)

        # logits: (batch, Lw, Lc, vocab_size)
        # stop_logits: (batch, Lw + 1)
        # predictions: (batch, Lw, Lc)
        logits = mem.get_stacked_tensor('logit').reshape(batch_sz, num_word_step, num_char_step, -1)
        stop_logits = mem.get_concatenated_tensor('word_stop')
        predictions = logits.argmax(dim=-1)
        return predictions, logits, stop_logits

    def _get_num_steps(self, target, is_for_word: bool) -> int:
        """ :param target: (B, L_w, L_c) """
        if target is not None:
            return target.size()[1] if is_for_word else target.size()[2] - 1
        return self.max_word_step if is_for_word else self.max_char_step

    def _forward_char_loop(self, start_tok, target, word_step, char_hx, char_attn_fn, mem):
        last_pred = start_tok
        for char_step in range(self._get_num_steps(target, is_for_word=False)):
            step_input = self._choose_rnn_input(last_pred, None if target is None else target[:, word_step, char_step])
            step_emb = self._get_step_embedding(step_input)
            char_cell_in = self._get_cell_input(step_emb, char_hx, char_attn_fn, is_for_word=False)
            char_hx, char_cell_out = self.char_decoder(char_cell_in, char_hx)
            proj_inp = self._get_proj_input(char_cell_out, char_attn_fn)
            char_logit = self._get_step_projection(proj_inp)
            last_pred = torch.argmax(char_logit, dim=-1)
            mem(logit=char_logit)

    def _choose_rnn_input(self, last_pred, last_gold: Optional):
        if self.training and np.random.rand(1).item() < self.scheduled_sampling_ratio:
            # use self-predicted tokens for scheduled sampling in training with _scheduled_sampling_ratio
            # step_inputs: (batch,)
            step_inputs = last_pred
        elif not self.training or last_gold is None:
            # no target present, maybe in validation
            # step_inputs: (batch,)
            step_inputs = last_pred
        else:
            # gold choice
            # step_inputs: (batch,)
            step_inputs = last_gold
        return step_inputs

    def _get_step_embedding(self, step_inp):
        # inputs_embedding: (batch, embedding_dim)
        inputs_embedding = self.tgt_emb_layer(step_inp)
        inputs_embedding = self.tgt_emb_dropout(inputs_embedding)
        return inputs_embedding

    def _get_cell_input(self, inputs_embedding, hx, enc_attn_fn, is_for_word):
        # compute attention context before the output is updated
        # actually only the first layer of decoder is using attention,
        # and only the final output of encoder and decoder history are attended over, not inter-layer states
        if is_for_word:
            decoder = self.word_decoder
            attn_comp = self.word_dec_inp_attn_comp
        else:
            decoder = self.char_decoder
            attn_comp = self.char_dec_inp_attn_comp

        prev_output = decoder.get_output_state(hx)
        prev_context = enc_attn_fn(prev_output)
        cell_inp = attn_comp(prev_context, inputs_embedding)
        return cell_inp

    def _get_proj_input(self, cell_out, enc_attn_fn):
        step_context = enc_attn_fn(cell_out)
        proj_inp = self.proj_inp_attn_comp(cell_out, step_context)
        return proj_inp

    def _get_step_projection(self, proj_input):
        proj_input = self.proj_inp_dropout(proj_input)
        step_logit = self.word_proj_layer(proj_input)
        return step_logit

    def _compute_metrics(self, source_mask, predictions, logits, target, target_mask):
        # gold, gold_mask: (B, Lw, Lc)
        gold = target[:, :, 1:].contiguous()
        gold_mask = target_mask[:, :, 1:].contiguous()
        # self.bleu(predictions, gold)
        # batch_xent: (batch,)
        batch_xent = seq_cross_ent(logits, gold, gold_mask, average=None)
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

    @classmethod
    def from_param_and_vocab(cls, p, vocab: NSVocabulary):
        """
        Build the seq2seq model with hyper-parameters.
        Since S2S is widely used and can be implemented task-agnostic,
        the builder for trialbot is provided as default.

        p.emb_sz = 256
        p.src_namespace = 'ns_q'
        p.tgt_namespace = 'ns_lf'
        p.hidden_sz = 128

        p.encoder = "bilstm"
        p.num_enc_layers = 2
        p.enc_out_dim = p.hidden_sz

        p.dec_in_dim = p.hidden_sz # by default
        p.dec_out_dim = p.hidden_sz # by default
        p.proj_in_dim = p.hidden_sz # by default
        p.word_enc_attn = "none"
        p.char_enc_attn = "bilinear"
        p.word_dec_inp_composer = 'none'
        p.char_dec_inp_composer = 'cat_mapping'
        p.dec_inp_comp_activation = 'mish'
        p.proj_inp_composer = 'cat_mapping'
        p.proj_inp_comp_activation = 'mish'

        p.dropout = .2
        p.enc_dropout = p.dropout # by default
        p.dec_dropout = p.dropout # by default

        p.decoder = "lstm"
        p.num_dec_layers = 2
        p.max_word_step = 50
        p.max_char_step = 10

        p.scheduled_sampling = .1
        p.decoder_init_strategy = "forward_last_parallel"
        p.tied_decoder_embedding = True
        p.src_emb_pretrained_file = "~/.glove/glove.6B.100d.txt.gz"
        """
        from trialbot.data import START_SYMBOL, END_SYMBOL
        from torch import nn
        from models.base_s2s.stacked_rnn_cell import StackedRNNCell
        from models.base_s2s.stacked_encoder import StackedEncoder
        from ..modules.attention_wrapper import get_wrapped_attention
        from ..modules.attention_composer import get_attn_composer

        emb_sz = p.emb_sz
        src_pretrain_file = getattr(p, 'src_emb_pretrained_file', None)
        if src_pretrain_file is None:
            source_embedding = nn.Embedding(num_embeddings=vocab.get_vocab_size(p.src_namespace), embedding_dim=emb_sz)
        else:
            from allennlp.modules.token_embedders import Embedding
            source_embedding = Embedding(embedding_dim=emb_sz,
                                         num_embeddings=vocab.get_vocab_size(p.src_namespace),
                                         vocab_namespace=p.src_namespace,
                                         pretrained_file=src_pretrain_file,
                                         vocab=vocab)
        if p.src_namespace == p.tgt_namespace:
            target_embedding = source_embedding
        else:
            target_embedding = nn.Embedding(num_embeddings=vocab.get_vocab_size(p.tgt_namespace), embedding_dim=emb_sz)

        encoder = StackedEncoder.get_encoder(p)
        enc_out_dim = encoder.get_output_dim()
        dec_in_dim = getattr(p, 'dec_in_dim', p.hidden_sz)
        dec_out_dim = getattr(p, 'dec_out_dim', p.hidden_sz)
        proj_in_dim = getattr(p, 'proj_in_dim', p.hidden_sz)

        # dec output attend over encoder outputs, thus attn_context dimension == enc_output_dim
        word_enc_attn = get_wrapped_attention(getattr(p, 'word_enc_attn', p.enc_attn), dec_out_dim, enc_out_dim)
        char_enc_attn = get_wrapped_attention(getattr(p, 'char_enc_attn', p.enc_attn), dec_out_dim, enc_out_dim)

        attn_sz = enc_out_dim
        word_dec_inp_composer = get_attn_composer(p.word_dec_inp_composer, attn_sz, dec_out_dim, dec_in_dim, p.dec_inp_comp_activation)
        # a char input composer will use hidden states for attention, but embeddings to compose
        char_dec_inp_composer = get_attn_composer(p.char_dec_inp_composer, attn_sz, emb_sz, dec_in_dim, p.dec_inp_comp_activation)
        proj_inp_composer = get_attn_composer(p.proj_inp_composer, attn_sz, dec_out_dim, proj_in_dim, p.proj_inp_comp_activation)

        enc_dropout = getattr(p, 'enc_dropout', p.dropout)
        dec_dropout = getattr(p, 'dec_dropout', p.dropout)
        rnn_cls = BaseSeq2Seq.get_decoder_type(p.decoder)
        word_decoder = StackedRNNCell([
            rnn_cls(dec_in_dim if floor == 0 else dec_out_dim, dec_out_dim)
            for floor in range(p.num_dec_layers)
        ], dec_dropout)
        char_decoder = StackedRNNCell([
            rnn_cls(dec_in_dim if floor == 0 else dec_out_dim, dec_out_dim)
            for floor in range(p.num_dec_layers)
        ], dec_dropout)

        word_proj = nn.Linear(proj_in_dim, vocab.get_vocab_size(p.tgt_namespace))
        if p.tied_decoder_embedding:
            assert proj_in_dim == emb_sz, f"Tied embeddings must have the same dimensions, proj{proj_in_dim} != emb{emb_sz}"
            word_proj.weight = target_embedding.weight  # tied embedding

        seq_pred = nn.Sequential(nn.Linear(dec_out_dim, 1), nn.Sigmoid())

        model = Seq2HierSeq(
            vocab=vocab,
            source_embedding=source_embedding,
            target_embedding=target_embedding,
            encoder=encoder,
            word_decoder=word_decoder,
            char_decoder=char_decoder,
            word_projector=word_proj,
            seq_len_predictor=seq_pred,
            enc_attn_word=word_enc_attn,
            enc_attn_char=char_enc_attn,
            word_dec_inp_attn_comp=word_dec_inp_composer,
            char_dec_inp_attn_comp=char_dec_inp_composer,
            proj_inp_attn_comp=proj_inp_composer,
            source_namespace=p.src_namespace,
            target_namespace=p.tgt_namespace,
            start_symbol=START_SYMBOL,
            eos_symbol=END_SYMBOL,
            padding_index=0,
            max_word_step=getattr(p, 'max_word_step', 50),
            max_char_step=getattr(p, 'max_char_step', 10),
            word_decoder_init_strategy=p.decoder_init_strategy,
            scheduled_sampling_ratio=p.scheduled_sampling,
            enc_dropout=enc_dropout,
            dec_dropout=dec_dropout,
        )
        return model
