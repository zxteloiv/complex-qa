from typing import Union, Optional, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from trialbot.data.ns_vocabulary import NSVocabulary
from models.transformer.multi_head_attention import SingleTokenMHAttentionWrapper
from models.modules.stacked_rnn_cell import StackedRNNCell
from utils.nn import filter_cat, filter_sum, prepare_input_mask, seq_cross_ent, get_decoder_initial_states
from utils.nn import AllenNLPAttentionWrapper
from utils.seq_collector import SeqCollector
from utils.text_tool import make_human_readable_text
from allennlp.training.metrics import BLEU, Perplexity, Average
from models.modules.stacked_encoder import StackedEncoder

class BaseSeq2Seq(torch.nn.Module):
    def __init__(self,
                 vocab: NSVocabulary,
                 encoder: StackedEncoder,
                 decoder: StackedRNNCell,
                 word_projection: torch.nn.Module,
                 source_embedding: torch.nn.Embedding,
                 target_embedding: torch.nn.Embedding,
                 source_namespace: str = "source_tokens",
                 target_namespace: str = "target_tokens",
                 start_symbol: str = '<GO>',
                 eos_symbol: str = '<EOS>',
                 max_decoding_step: int = 50,
                 enc_attention: Union[AllenNLPAttentionWrapper, SingleTokenMHAttentionWrapper, None] = None,
                 dec_hist_attn: Union[AllenNLPAttentionWrapper, SingleTokenMHAttentionWrapper, None] = None,
                 scheduled_sampling_ratio: float = 0.,
                 intermediate_dropout: float = .1,
                 concat_attn_to_dec_input: bool = False,
                 padding_index: int = 0,
                 decoder_init_strategy: str = "forward_all"
                 ):
        super().__init__()
        self.vocab = vocab
        self._enc_attn = enc_attention
        self._dec_hist_attn = dec_hist_attn

        self._scheduled_sampling_ratio = scheduled_sampling_ratio
        self._encoder = encoder
        self._decoder = decoder
        self._src_embedding = source_embedding
        self._tgt_embedding = target_embedding

        src_hidden_dim = self._encoder.get_output_dim()
        tgt_hidden_dim = self._decoder.hidden_dim

        self._enc_attn_mapping = torch.nn.Linear(src_hidden_dim, tgt_hidden_dim)
        self._dec_hist_attn_mapping = torch.nn.Linear(tgt_hidden_dim, tgt_hidden_dim)

        self._start_id = vocab.get_token_index(start_symbol, target_namespace)
        self._eos_id = vocab.get_token_index(eos_symbol, target_namespace)
        self._max_decoding_step = max_decoding_step

        self._source_namespace = source_namespace
        self._target_namespace = target_namespace

        self._output_projection = word_projection

        self._concat_attn = concat_attn_to_dec_input
        self._dropout = torch.nn.Dropout(intermediate_dropout)

        self._padding_index = padding_index
        self._strategy = decoder_init_strategy

        self.bleu = BLEU(exclude_indices={padding_index, self._start_id, self._eos_id})
        self.ppl = Perplexity()
        self.err_rate = Average()

    def forward(self,
                source_tokens: torch.LongTensor,
                target_tokens: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        """Run the network, and dispatch work to helper functions based on the runtime"""

        # source: (batch, source_length), containing the input word IDs
        # target: (batch, target_length), containing the output IDs

        source, source_mask = prepare_input_mask(source_tokens, self._padding_index)
        state, layer_states = self._encode(source, source_mask)
        layer_initial = get_decoder_initial_states(layer_states, source_mask, self._strategy,
                                                   self._encoder.is_bidirectional(), self._decoder.get_layer_num())
        init_hidden, _ = self._decoder.init_hidden_states(layer_initial)

        output = {"source": source}
        # Optional[torch.LongTensor] (batch, seq_len)
        target, target_mask = prepare_input_mask(target_tokens, self._padding_index)
        if self.training:
            assert target is not None
            # predictions: (batch, seq_len)
            # logits: (batch, seq_len, vocab_size)
            predictions, logits = self._forward_loop(state, source_mask, init_hidden, target, target_mask)
            loss = seq_cross_ent(logits, target[:, 1:].contiguous(), target_mask[:, 1:].contiguous())
            output['loss'] = loss
        else:
            predictions, logits = self._forward_loop(state, source_mask, init_hidden, None, None)

        output.update(predictions=predictions, logits=logits, target=target)
        if target is not None:
            self.compute_metrics(predictions, logits, target, target_mask)

        return output

    def compute_metrics(self, predictions, logits, target, target_mask):
        # gold, gold_mask: (batch, max_tgt_len - 1)
        gold = target[:, 1:].contiguous()
        gold_mask = target_mask[:, 1:].contiguous()
        self.bleu(predictions, gold)
        # batch_xent: (batch,)
        batch_xent = seq_cross_ent(logits, gold, gold_mask, average=None)
        for xent in batch_xent:
            self.ppl(xent)

        total_err = ((predictions != gold) * gold_mask).sum(list(range(gold_mask.ndim))[1:]) > 0
        for instance_err in total_err:
            self.err_rate(instance_err)

    def get_metric(self, reset=False):
        metric = {"PPL": self.ppl.get_metric(reset), "ERR": self.err_rate.get_metric(reset)}
        metric.update(self.bleu.get_metric(reset))
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
            output_dict['target'], self.vocab, self._target_namespace,
            stop_ids=[self._eos_id, self._padding_index]
        )
        return output_dict

    def _encode(self,
                source: torch.LongTensor,
                source_mask: torch.LongTensor):
        # source: (batch, max_input_length), source sequence token ids
        # source_mask: (batch, max_input_length), source sequence padding mask
        # source_embedding: (batch, max_input_length, embedding_sz)
        source_embedding = self._src_embedding(source)
        source_embedding = self._dropout(source_embedding)
        source_hidden, layered_hidden = self._encoder(source_embedding, source_mask)
        return source_hidden, layered_hidden

    def _forward_loop(self,
                      source_state: torch.Tensor,
                      source_mask: Optional[torch.LongTensor],
                      init_hidden: torch.Tensor,
                      target: Optional[torch.LongTensor],
                      target_mask: Optional[torch.LongTensor],
                      ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Do the decoding process for training and prediction

        :param source_state: (batch, max_input_length, hidden_dim),
        :param source_mask: (batch, max_input_length)
        :param target: (batch, max_target_length)
        :param target_mask: (batch, max_target_length)
        :return:
        """

        # shape: (batch, max_input_sequence_length)
        batch = source_state.size()[0]

        if target is not None:
            num_decoding_steps = target.size()[1] - 1
        else:
            num_decoding_steps = self._max_decoding_step

        # Initialize target predictions with the start index.
        # batch_start: (batch_size,)
        batch_start = source_mask.new_full((batch,), fill_value=self._start_id)

        if self._enc_attn is not None:
            enc_attn_fn = lambda out: self._enc_attn_mapping(self._enc_attn(out, source_state, source_mask))
        else:
            enc_attn_fn = None

        mem = SeqCollector()
        last_pred = batch_start
        for timestep in range(num_decoding_steps):
            step_input = self._choose_rnn_input(last_pred, None if target is None else target[:, timestep])
            # inputs_embedding: (batch, embedding_dim)
            inputs_embedding = self._tgt_embedding(step_input)
            inputs_embedding = self._dropout(inputs_embedding)

            # build runtime decoder history attention module.
            dec_hist_attn_fn = self._create_runtime_dec_hist_attn_fn(mem, target_mask, timestep)

            dec_out = self._forward_step(inputs_embedding, init_hidden, enc_attn_fn, dec_hist_attn_fn)

            init_hidden, step_output, step_logit = dec_out[:3]
            # greedy decoding
            # last_pred: (batch, )
            last_pred = torch.argmax(step_logit, dim=-1)
            mem(output=step_output, logit=step_logit)

        # logits: (batch, seq_len, vocab_size)
        # predictions: (batch, seq_len)
        logits = mem.get_stacked_tensor('logit')
        predictions = logits.argmax(dim=-1)

        return predictions, logits

    def _choose_rnn_input(self, last_pred, last_gold: Optional):
        if self.training and np.random.rand(1).item() < self._scheduled_sampling_ratio:
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

    def _create_runtime_dec_hist_attn_fn(self, mem: SeqCollector, target_mask, timestep: int):
        # build runtime decoder history attention module.
        if self._dec_hist_attn is None:
            dec_hist_attn_fn = None

        elif timestep > 0:  # which suggest that output_by_step is not empty
            dec_hist = mem.get_stacked_tensor('output')
            dec_hist_mask = target_mask[:, :timestep] if target_mask is not None else None
            dec_hist_attn_fn = lambda out: self._dec_hist_attn_mapping(
                self._dec_hist_attn(out, dec_hist, dec_hist_mask)
            )

        else:
            dec_hist_attn_fn = lambda out: torch.zeros_like(out)

        return dec_hist_attn_fn

    def _forward_step(self, inputs_embedding, step_hidden, enc_attn_fn, dec_hist_attn_fn):
        # compute attention context before the output is updated
        # actually only the first layer of decoder is using attention,
        # and only the final output of encoder and decoder history are attended over, not inter-layer states
        prev_output = self._decoder.get_output_state(step_hidden)
        enc_context = enc_attn_fn(prev_output) if enc_attn_fn else None
        dec_hist_context = dec_hist_attn_fn(prev_output) if dec_hist_attn_fn else None

        # step_hidden: some_hidden_var_with_unknown_internals
        # step_output: (batch, hidden_dim)
        cat_context = []
        if self._concat_attn and enc_context is not None:
            cat_context.append(self._dropout(enc_context))
        if self._concat_attn and dec_hist_context is not None:
            cat_context.append(self._dropout(dec_hist_context))
        dec_output = self._decoder(inputs_embedding, step_hidden, cat_context)
        step_hidden, step_output = dec_output[:2]

        step_logit = self._get_step_projection(step_output, enc_context, dec_hist_context)
        return step_hidden, step_output, step_logit

    def _get_step_projection(self, *inputs):
        # step_logit: (batch, vocab_size)
        if self._concat_attn:
            proj_input = filter_cat(inputs, dim=-1)
        else:
            proj_input = filter_sum(inputs)

        proj_input = self._dropout(proj_input)
        step_logit = self._output_projection(proj_input)
        return step_logit

class BaseS2SBuilder:
    """
    Build the seq2seq model with hyper-parameters.
    Since S2S is widely used and can be implemented task-agnostic,
    the builder for trialbot is provided as default.

    p.emb_sz = 256
    p.src_namespace = 'ns_q'
    p.tgt_namespace = 'ns_lf'
    p.hidden_sz = 128
    p.enc_attn = "bilinear"
    p.dec_hist_attn = "dot_product"
    p.concat_attn_to_dec_input = True
    p.encoder = "bilstm"
    p.num_enc_layers = 2
    p.dropout = .2
    p.decoder = "lstm"
    p.num_dec_layers = 2
    p.max_decoding_step = 100
    p.scheduled_sampling = .1
    p.decoder_init_strategy = "forward_last_parallel"
    """
    @classmethod
    def from_param_and_vocab(cls, p, vocab: NSVocabulary):

        from trialbot.data import NSVocabulary, PADDING_TOKEN, START_SYMBOL, END_SYMBOL
        from torch import nn
        from models.modules.stacked_rnn_cell import StackedLSTMCell, StackedGRUCell, StackedRNNCell

        emb_sz = p.emb_sz
        source_embedding = nn.Embedding(num_embeddings=vocab.get_vocab_size(p.src_namespace), embedding_dim=emb_sz)
        target_embedding = nn.Embedding(num_embeddings=vocab.get_vocab_size(p.tgt_namespace), embedding_dim=emb_sz)

        encoder = cls._get_encoder(p)
        enc_out_dim = encoder.get_output_dim()
        dec_out_dim = p.hidden_sz  # the output dimension of decoder is just the same as hidden_dim

        # dec output attend over previous dec outputs, thus attn_context dimension == dec_output_dim
        dec_hist_attn = cls._get_attention(p.dec_hist_attn, dec_out_dim, dec_out_dim)
        # dec output attend over encoder outputs, thus attn_context dimension == enc_output_dim
        enc_attn = cls._get_attention(p.enc_attn, dec_out_dim, enc_out_dim)

        if p.enc_attn == 'dot_product':
            assert enc_out_dim == dec_out_dim, "encoder hidden states must be able to multiply with decoder output"

        def sum_attn_dims(attns, dims):
            """Compute the dimension requirements for all attention modules"""
            return sum(dim for attn, dim in zip(attns, dims) if attn is not None)

        # since attentions are mapped into hidden size, the hidden_dim is used instead of original context size
        attn_size = sum_attn_dims([enc_attn, dec_hist_attn], [dec_out_dim, dec_out_dim])

        dec_in_dim = emb_sz + (attn_size if p.concat_attn_to_dec_input else 0)
        rnn_cls = cls._get_decoder_type(p)
        decoder = StackedRNNCell(rnn_cls, dec_in_dim, dec_out_dim, p.num_dec_layers, p.dropout)

        proj_in_dim = dec_out_dim + (attn_size if p.concat_attn_to_dec_input else 0)
        word_proj = nn.Linear(proj_in_dim, vocab.get_vocab_size(p.tgt_namespace))
        if p.tied_decoder_embedding:
            word_proj.weight = target_embedding.weight  # tied embedding

        model = BaseSeq2Seq(
            vocab=vocab,
            encoder=encoder,
            decoder=decoder,
            word_projection=word_proj,
            source_embedding=source_embedding,
            target_embedding=target_embedding,
            source_namespace=p.src_namespace,
            target_namespace=p.tgt_namespace,
            start_symbol=START_SYMBOL,
            eos_symbol=END_SYMBOL,
            max_decoding_step=p.max_decoding_step,
            enc_attention=enc_attn,
            dec_hist_attn=dec_hist_attn,
            scheduled_sampling_ratio=p.scheduled_sampling,
            intermediate_dropout=p.dropout,
            concat_attn_to_dec_input=p.concat_attn_to_dec_input,
            padding_index=0,
            decoder_init_strategy=p.decoder_init_strategy,
        )
        return model

    @staticmethod
    def _get_encoder(p):
        from models.modules.stacked_encoder import StackedEncoder
        from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
        from models.transformer.encoder import TransformerEncoder
        emb_sz, hidden_dim = p.emb_sz, p.hidden_sz

        if p.encoder == 'lstm':
            encoder = StackedEncoder([
                PytorchSeq2SeqWrapper(nn.LSTM(emb_sz if floor == 0 else hidden_dim, hidden_dim, batch_first=True))
                for floor in range(p.num_enc_layers)
            ], emb_sz, hidden_dim, input_dropout=p.dropout)
        elif p.encoder == 'bilstm':
            encoder = StackedEncoder([
                PytorchSeq2SeqWrapper(nn.LSTM(emb_sz if floor == 0 else hidden_dim * 2,
                                              hidden_dim, batch_first=True, bidirectional=True))
                for floor in range(p.num_enc_layers)
            ], emb_sz, hidden_dim, input_dropout=p.dropout)
        elif p.encoder == 'transformer':
            encoder = StackedEncoder([
                TransformerEncoder(input_dim=emb_sz if floor == 0 else hidden_dim,
                                   hidden_dim=hidden_dim,
                                   num_layers=1,
                                   num_heads=p.num_heads,
                                   feedforward_hidden_dim=hidden_dim,
                                   feedforward_dropout=p.dropout,
                                   residual_dropout=p.dropout,
                                   attention_dropout=p.attention_dropout,
                                   use_positional_embedding=(floor == 0)
                                   )
                for floor in range(p.num_enc_layers)
            ], emb_sz, hidden_dim, input_dropout=p.dropout)
        else:
            assert False

        return encoder

    @staticmethod
    def _get_attention(attn_type: str, vector_dim: int, matrix_dim: int, attention_dropout: float = 0., **kwargs):
        """
        Build an Attention module with specified parameters.
        :param attn_type: indicates the attention type, e.g. "bilinear", "dot_product" or "none"
        :param vector_dim: the vector to compute attention
        :param matrix_dim: the bunch of vectors to attend against (batch, num, matrix_dim)
        :param attention_dropout: the dropout to discard some attention weights
        :return: a torch.nn.Module
        """
        from utils.nn import AllenNLPAttentionWrapper
        from allennlp.modules.attention import BilinearAttention, DotProductAttention

        attn_type = attn_type.lower()
        if attn_type == "bilinear":
            attn = BilinearAttention(vector_dim=vector_dim, matrix_dim=matrix_dim)
            attn = AllenNLPAttentionWrapper(attn, attention_dropout)
        elif attn_type == "dot_product":
            attn = DotProductAttention()
            attn = AllenNLPAttentionWrapper(attn, attention_dropout)
        elif attn_type == "none":
            attn = None
        else:
            assert False

        return attn

    @staticmethod
    def _get_decoder_type(p):
        from models.modules.universal_hidden_state_wrapper import UniversalHiddenStateWrapper, RNNType
        cell_type = p.decoder
        if cell_type == "lstm":
            cls = RNNType.LSTM
        elif cell_type == "gru":
            cls = RNNType.GRU
        elif cell_type == "ind_rnn":
            cls = RNNType.IndRNN
        elif cell_type == "rnn":
            cls = RNNType.VanillaRNN
        else:
            raise ValueError(f"RNN type of {cell_type} not found.")

        return cls



