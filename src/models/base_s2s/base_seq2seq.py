from typing import Union, Optional, Tuple, Dict, Any, Literal
import numpy as np
import torch
from torch import nn
from trialbot.data.ns_vocabulary import NSVocabulary
from ..interfaces.attention import Attention as IAttn, VectorContextComposer as AttnComposer
from ..modules.variational_dropout import VariationalDropout
from ..interfaces.unified_rnn import RNNStack, EncoderRNNStack
from .stacked_rnn_cell import StackedRNNCell
from utils.nn import filter_cat, prepare_input_mask, seq_cross_ent
from utils.nn import aggregate_layered_state, assign_stacked_states
from utils.seq_collector import SeqCollector
from utils.text_tool import make_human_readable_text
from allennlp.training.metrics import BLEU, Perplexity, Average


class BaseSeq2Seq(torch.nn.Module):
    def __init__(self,
                 # modules
                 vocab: NSVocabulary,
                 encoder: EncoderRNNStack,
                 decoder: RNNStack,
                 word_projection: torch.nn.Module,
                 source_embedding: torch.nn.Embedding,
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
                 decoder_init_strategy: str = "forward_all",
                 enc_dec_transform_usage: str = 'consistent',

                 # training_configuration
                 scheduled_sampling_ratio: float = 0.,
                 enc_dropout: float = 0,
                 dec_dropout: float = .1,
                 training_average: Literal["token", "batch", "none"] = "batch",
                 ):
        super().__init__()
        self.vocab = vocab
        self._enc_attn = enc_attention
        self._dec_hist_attn = dec_hist_attn

        self._dec_inp_attn_comp = dec_inp_attn_comp
        self._proj_inp_attn_comp = proj_inp_attn_comp

        self._scheduled_sampling_ratio = scheduled_sampling_ratio
        self._encoder = encoder
        self._decoder = decoder
        self._src_embedding = source_embedding
        self._tgt_embedding = target_embedding
        self._enc_dec_trans = enc_dec_transformer
        self._enc_dec_trans_usage = enc_dec_transform_usage

        self._start_id = vocab.get_token_index(start_symbol, target_namespace)
        self._eos_id = vocab.get_token_index(eos_symbol, target_namespace)
        self._max_decoding_step = max_decoding_step

        self._source_namespace = source_namespace
        self._target_namespace = target_namespace

        self._output_projection = word_projection

        self._src_emb_dropout = VariationalDropout(enc_dropout, on_the_fly=True)
        self._tgt_emb_dropout = VariationalDropout(dec_dropout, on_the_fly=False)
        self._proj_inp_dropout = VariationalDropout(dec_dropout, on_the_fly=False)

        self._padding_index = padding_index
        self._strategy = decoder_init_strategy
        self._training_avg = training_average

        self.bleu = BLEU(exclude_indices={padding_index, self._start_id, self._eos_id})
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

    def forward(self, source_tokens: torch.LongTensor, target_tokens: torch.LongTensor = None) -> Dict[
        str, torch.Tensor]:
        # source: (batch, source_length), containing the input word IDs
        # target: (batch, target_length), containing the output IDs
        source, source_mask = prepare_input_mask(source_tokens, self._padding_index)
        target, target_mask = prepare_input_mask(target_tokens, self._padding_index)
        output = {"source": source}

        self._reset_variational_dropouts()

        state, layer_states = self._encode(source, source_mask)
        hx = self._init_decoder(layer_states, source_mask)
        enc_attn_fn = self._get_enc_attn_fn(state, source_mask)

        if self.training:
            assert target is not None
            # preds: (batch, seq_len)
            # logits: (batch, seq_len, vocab_size)
            preds, logits = self._forward_dec_loop(source_mask, enc_attn_fn, hx, target, target_mask)
            output['loss'] = seq_cross_ent(logits, target[:, 1:].contiguous(), target_mask[:, 1:].contiguous(),
                                           self._training_avg)
        else:
            runtime_len = -1 if target is None else target.size()[1] - 1
            preds, logits = self._forward_dec_loop(source_mask, enc_attn_fn, hx, None, None, runtime_len)

        output.update(predictions=preds, logits=logits, target=target)

        if target is not None:
            total_err = self._compute_metrics(source_mask, preds, logits, target, target_mask)
            output.update(errno=total_err.tolist())

        return output

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

    def _reset_variational_dropouts(self):
        for m in self.modules():
            if isinstance(m, VariationalDropout):
                m.reset()

    def _encode(self, source: torch.LongTensor, source_mask: torch.LongTensor):
        # source: (batch, max_input_length), source sequence token ids
        # source_mask: (batch, max_input_length), source sequence padding mask
        # source_embedding: (batch, max_input_length, embedding_sz)
        source_embedding = self._src_embedding(source)
        source_embedding = self._src_emb_dropout(source_embedding)
        source_hidden, layered_hidden = self._encoder(source_embedding, source_mask)
        if self._enc_dec_trans_usage == 'consistent':
            source_hidden = self._enc_dec_trans(source_hidden)
            layered_hidden = list(map(self._enc_dec_trans, layered_hidden))
        return source_hidden, layered_hidden

    def _init_decoder(self, layer_states, source_mask):
        usage: str = self._enc_dec_trans_usage
        use_first_half: bool = self._encoder.is_bidirectional() and usage not in ('dec_init', 'consistent')

        agg_src = aggregate_layered_state(layer_states, source_mask, self._strategy, use_first_half)
        if usage == 'dec_init':
            agg_src = list(map(self._enc_dec_trans, agg_src))

        dec_states = assign_stacked_states(agg_src, self._decoder.get_layer_num(), self._strategy)

        # init the internal hiddens
        hx, _ = self._decoder.init_hidden_states(dec_states)
        return hx

    def _forward_dec_loop(self, source_mask: torch.Tensor, enc_attn_fn, hx: Any,
                          target: Optional[torch.LongTensor], target_mask: Optional[torch.LongTensor],
                          runtime_max_decoding_len: int = -1,
                          ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        :param enc_attn_fn: tensor -> tensor
        :param hx:
        :param target:
        :param target_mask:
        :param runtime_max_decoding_len:
        :return:
        """
        last_pred = source_mask.new_full((source_mask.size()[0],), fill_value=self._start_id)
        num_decoding_steps = self._get_decoding_loop_len(target, runtime_max_decoding_len)
        mem = SeqCollector()
        for timestep in range(num_decoding_steps):
            step_input = self._choose_rnn_input(last_pred, None if target is None else target[:, timestep])
            inputs_embedding = self._get_step_embedding(step_input)
            dec_hist_attn_fn = self._create_runtime_dec_hist_attn_fn(mem, target_mask, timestep)
            cell_inp = self._get_cell_input(inputs_embedding, hx, enc_attn_fn, dec_hist_attn_fn)
            hx, cell_out = self._decoder(cell_inp, hx)
            proj_inp = self._get_proj_input(cell_out, enc_attn_fn, dec_hist_attn_fn)
            step_logit = self._get_step_projection(proj_inp)
            # greedy decoding
            # last_pred: (batch,)
            last_pred = torch.argmax(step_logit, dim=-1)
            mem(output=cell_out, logit=step_logit)

        # logits: (batch, seq_len, vocab_size)
        # predictions: (batch, seq_len)
        logits = mem.get_stacked_tensor('logit')
        predictions = logits.argmax(dim=-1)

        return predictions, logits

    def _get_enc_attn_fn(self, source_state, source_mask):
        if 'attn' == self._enc_dec_trans_usage:
            source_state = self._enc_dec_trans(source_state)

        def enc_attn_fn(out):
            if self._enc_attn is None:
                return None
            else:
                return self._enc_attn(out, source_state, source_mask)

        return enc_attn_fn

    def _get_decoding_loop_len(self, target, maximum):
        if target is not None:
            num_decoding_steps = target.size()[1] - 1
        elif maximum > 0:
            num_decoding_steps = maximum
        else:
            num_decoding_steps = self._max_decoding_step
        return num_decoding_steps

    def _choose_rnn_input(self, last_pred, last_gold: Optional):
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

    def _get_step_embedding(self, step_inp):
        # inputs_embedding: (batch, embedding_dim)
        inputs_embedding = self._tgt_embedding(step_inp)
        inputs_embedding = self._tgt_emb_dropout(inputs_embedding)
        return inputs_embedding

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

    def _get_cell_input(self, inputs_embedding, hx, enc_attn_fn, dec_hist_attn_fn):
        # compute attention context before the output is updated
        # actually only the first layer of decoder is using attention,
        # and only the final output of encoder and decoder history are attended over, not inter-layer states
        prev_output = self._decoder.get_output_state(hx)
        prev_enc_context = enc_attn_fn(prev_output)
        prev_dec_context = dec_hist_attn_fn(prev_output)

        prev_context = filter_cat([prev_enc_context, prev_dec_context], dim=-1)
        cell_inp = self._dec_inp_attn_comp(prev_context, inputs_embedding)
        return cell_inp

    def _get_proj_input(self, cell_out, enc_attn_fn, dec_hist_attn_fn):
        step_enc_context = enc_attn_fn(cell_out)
        step_dec_context = dec_hist_attn_fn(cell_out)
        step_context = filter_cat([step_enc_context, step_dec_context], dim=-1)
        proj_inp = self._proj_inp_attn_comp(cell_out, step_context)
        return proj_inp

    def _get_step_projection(self, proj_input):
        proj_input = self._proj_inp_dropout(proj_input)
        step_logit = self._output_projection(proj_input)
        return step_logit

    def _compute_metrics(self, source_mask, predictions, logits, target, target_mask):
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

    @classmethod
    def from_param_and_vocab(cls, p, vocab: NSVocabulary):
        """
        Build the seq2seq model with hyper-parameters.
        Since S2S is widely used and can be implemented task-agnostic,
        the builder for trialbot is provided as default.
        Because we want a flat configuration file and use it only with the BaseSeq2Seq model,
        the model is responsible for providing definitions.

        p.emb_sz = 256
        p.src_namespace = 'ns_q'
        p.tgt_namespace = 'ns_lf'
        p.hidden_sz = 128
        p.dec_in_dim = p.hidden_sz # by default
        p.dec_out_dim = p.hidden_sz # by default
        p.proj_in_dim = p.hidden_sz # by default
        p.enc_attn = "bilinear"
        p.dec_hist_attn = "dot_product"
        p.dec_inp_composer = 'cat_mapping'
        p.dec_inp_comp_activation = 'mish'
        p.proj_inp_composer = 'cat_mapping'
        p.proj_inp_comp_activation = 'mish'
        p.enc_dec_trans_act = 'linear'
        p.enc_dec_trans_usage = 'consistent'
        p.enc_dec_trans_forced = True
        p.use_cell_based_encoder = False
        p.encoder = "bilstm"
        p.cell_encoder_is_bidirectional = False
        p.num_enc_layers = 2
        p.decoder = "lstm"
        p.num_dec_layers = 2
        p.dropout = .2
        p.enc_dropout = p.dropout # by default
        p.dec_dropout = p.dropout # by default
        p.max_decoding_step = 100
        p.scheduled_sampling = .1
        p.decoder_init_strategy = "forward_last_parallel"
        p.tied_decoder_embedding = True
        p.src_emb_pretrained_file = "~/.glove/glove.6B.100d.txt.gz"
        """
        from trialbot.data import START_SYMBOL, END_SYMBOL
        from ..modules.attention_wrapper import get_wrapped_attention
        from ..modules.attention_composer import get_attn_composer

        emb_sz = p.emb_sz
        source_embedding, target_embedding = cls._get_embeddings(p, vocab)

        encoder = cls.get_encoder(p)
        enc_out_dim = encoder.get_output_dim()
        dec_in_dim = getattr(p, 'dec_in_dim', p.hidden_sz)
        dec_out_dim = getattr(p, 'dec_out_dim', p.hidden_sz)
        proj_in_dim = getattr(p, 'proj_in_dim', p.hidden_sz)

        trans_module, _usages = cls._get_transformation_module(p, encoder.is_bidirectional(), enc_out_dim, dec_out_dim)
        trans_usage_string = cls._get_trans_usage_string(p, trans_module, _usages)

        # Initialize attentions. And compute the dimension requirements for all attention modules
        # the encoder attention size depends on the transformation usage
        enc_attn_sz = dec_out_dim if trans_usage_string in ('consistent', 'attn') else enc_out_dim
        enc_attn = get_wrapped_attention(p.enc_attn, dec_out_dim, enc_attn_sz)
        # dec output attend over previous dec outputs, thus attn_context dimension == dec_output_dim
        dec_hist_attn = get_wrapped_attention(p.dec_hist_attn, dec_out_dim, dec_out_dim)
        attn_sz = 0 if dec_hist_attn is None else dec_out_dim
        attn_sz += 0 if enc_attn is None else enc_attn_sz
        dec_inp_composer = get_attn_composer(p.dec_inp_composer, attn_sz, emb_sz, dec_in_dim, p.dec_inp_comp_activation)
        proj_inp_composer = get_attn_composer(p.proj_inp_composer, attn_sz, dec_out_dim, proj_in_dim,
                                              p.proj_inp_comp_activation)
        if enc_attn is not None or dec_hist_attn is not None:
            assert dec_inp_composer is not None and proj_inp_composer is not None, "Attention must be composed"

        enc_dropout = getattr(p, 'enc_dropout', getattr(p, 'dropout', 0.))
        dec_dropout = getattr(p, 'dec_dropout', getattr(p, 'dropout', 0.))
        decoder = cls.get_stacked_rnn(p.decoder, dec_in_dim, dec_out_dim, p.num_dec_layers,
                                      dec_dropout, dec_dropout)

        word_proj = nn.Linear(proj_in_dim, vocab.get_vocab_size(p.tgt_namespace))
        if p.tied_decoder_embedding:
            assert proj_in_dim == emb_sz, f"Tied embeddings must have the same dimensions, proj{proj_in_dim} != emb{emb_sz}"
            word_proj.weight = target_embedding.weight  # tied embedding

        model = BaseSeq2Seq(
            vocab=vocab,
            encoder=encoder,
            decoder=decoder,
            word_projection=word_proj,
            source_embedding=source_embedding,
            target_embedding=target_embedding,
            enc_attention=enc_attn,
            dec_hist_attn=dec_hist_attn,
            dec_inp_attn_comp=dec_inp_composer,
            proj_inp_attn_comp=proj_inp_composer,
            enc_dec_transformer=trans_module,
            source_namespace=p.src_namespace,
            target_namespace=p.tgt_namespace,
            start_symbol=START_SYMBOL,
            eos_symbol=END_SYMBOL,
            padding_index=0,
            max_decoding_step=p.max_decoding_step,
            decoder_init_strategy=p.decoder_init_strategy,
            enc_dec_transform_usage=trans_usage_string,
            scheduled_sampling_ratio=p.scheduled_sampling,
            enc_dropout=enc_dropout,
            dec_dropout=dec_dropout,
            training_average=getattr(p, "training_average", "batch"),
        )
        return model

    @staticmethod
    def _get_embeddings(p, vocab):
        emb_sz = p.emb_sz
        src_pretrain_file = getattr(p, 'src_emb_pretrained_file', None)
        if src_pretrain_file is None:
            source_embedding = nn.Embedding(num_embeddings=vocab.get_vocab_size(p.src_namespace),
                                            embedding_dim=emb_sz,
                                            padding_idx=0,
                                            )
        else:
            from allennlp.modules.token_embedders import Embedding
            source_embedding = Embedding(embedding_dim=emb_sz,
                                         num_embeddings=vocab.get_vocab_size(p.src_namespace),
                                         vocab_namespace=p.src_namespace,
                                         padding_index=0,
                                         pretrained_file=src_pretrain_file,
                                         vocab=vocab)
        if p.src_namespace == p.tgt_namespace:
            target_embedding = source_embedding
        else:
            target_embedding = nn.Embedding(num_embeddings=vocab.get_vocab_size(p.tgt_namespace), embedding_dim=emb_sz)

        return source_embedding, target_embedding

    @staticmethod
    def _get_transformation_module(p, enc_is_bidirectional: bool, enc_out_dim: int, dec_out_dim: int):
        import allennlp.nn
        # autodetect for the necessity of the transformation module
        # expecting the dimensions matchs between the decoder and encoder, otherwise a transformer is introduced.
        trans_for_dec_init = not (p.decoder_init_strategy.startswith('zero')
                                  or (enc_is_bidirectional and dec_out_dim * 2 == enc_out_dim)
                                  or (not enc_is_bidirectional and dec_out_dim == enc_out_dim))
        # expecting the encoder output and decoder output matches when attention is dot-product,
        # other attention types will transform the output dimension on their own.
        trans_for_attn = (p.enc_attn == 'dot_product' and enc_out_dim != dec_out_dim)
        # the transformer is configured to be
        forced_transformer: bool = getattr(p, 'enc_dec_trans_forced', False)
        enc_dec_transformer = None
        if trans_for_attn or trans_for_dec_init or forced_transformer:
            enc_dec_transformer = nn.Sequential(
                nn.Linear(enc_out_dim, dec_out_dim),
                allennlp.nn.Activation.by_name(getattr(p, 'enc_dec_trans_act', 'linear'))(),
            )
        return enc_dec_transformer, (trans_for_dec_init, trans_for_attn)

    @staticmethod
    def _get_trans_usage_string(p, trans_module, usage):
        used_by_dec_init, used_by_attn = usage
        # when transformer is present, check if the consistent transformation is preferred.
        # if it is so, a mere dec-init or attn usage will get overwritten
        # consistent will not work if the transformer is unavailable
        preferred_consistent: bool = getattr(p, 'enc_dec_trans_usage', 'consistent').lower() == 'consistent'
        if preferred_consistent and trans_module is not None:
            enc_dec_trans_usage = "consistent"  # the output will be transformed immediately and never in the future
        elif used_by_dec_init:
            enc_dec_trans_usage = "dec_init"    # the output will be only transformed when initializing the decoder
        elif used_by_attn:
            enc_dec_trans_usage = "attn"    # the output will be only transformed when computing attentions
        else:
            enc_dec_trans_usage = ""    # no need at all

        return enc_dec_trans_usage

    @staticmethod
    def get_encoder(p) -> EncoderRNNStack:

        use_cell_based_encoder = getattr(p, 'use_cell_based_encoder', False)
        if not use_cell_based_encoder:
            from .stacked_encoder import StackedEncoder
            return StackedEncoder.get_encoder(p)

        else:
            dropout = getattr(p, 'enc_dropout', getattr(p, 'dropout', 0.))
            hid_sz = getattr(p, 'enc_out_dim', p.hidden_sz)
            bid_cell = getattr(p, 'cell_encoder_is_bidirectional', False)
            stacked_cell = BaseSeq2Seq.get_stacked_rnn(p.encoder, p.emb_sz, hid_sz, p.num_enc_layers, dropout, dropout)
            back_cell = BaseSeq2Seq.get_stacked_rnn(p.encoder, p.emb_sz, hid_sz, p.num_enc_layers, dropout, dropout)
            from .stacked_cell_encoder import StackedCellEncoder
            return StackedCellEncoder(stacked_cell, backward_stacked_cell=None if not bid_cell else back_cell)

    @staticmethod
    def get_stacked_rnn(cell_type: str, inp_sz: int, hid_sz: int, num_layers: int,
                        h_dropout: float, v_dropout: float,
                        onlstm_chunk_sz: int = 10,
                        ) -> RNNStack:
        from .stacked_rnn_cell import StackedRNNCell
        from ..modules.torch_rnn_wrapper import TorchRNNWrapper as RNNWrapper
        from ..modules.variational_dropout import VariationalDropout

        def _get_cell_in(floor):
            return inp_sz if floor == 0 else hid_sz

        def _get_h_vd(d):
            return VariationalDropout(d, on_the_fly=False) if d > 0 else None

        rnns = cls = None
        if cell_type == 'typed_rnn':
            from ..modules.sym_typed_rnn_cell import SymTypedRNNCell
            rnns = [
                SymTypedRNNCell(_get_cell_in(floor), hid_sz, "tanh", _get_h_vd(h_dropout))
                for floor in range(num_layers)
            ]

        elif cell_type == 'onlstm':
            from ..onlstm.onlstm import ONLSTMCell
            rnns = [
                ONLSTMCell(_get_cell_in(floor), hid_sz, onlstm_chunk_sz, _get_h_vd(h_dropout))
                for floor in range(num_layers)
            ]

        elif cell_type == 'ind_rnn':
            from ..modules.independent_rnn import IndRNNCell
            rnns = [IndRNNCell(_get_cell_in(floor), hid_sz) for floor in range(num_layers)]

        elif cell_type == "lstm":
            cls = torch.nn.LSTMCell
        elif cell_type == "gru":
            cls = torch.nn.GRUCell
        elif cell_type == "rnn":
            cls = torch.nn.RNNCell
        else:
            raise ValueError(f"RNN type of {cell_type} not found.")

        if rnns is None:
            rnns = [RNNWrapper(cls(_get_cell_in(floor), hid_sz), _get_h_vd(h_dropout))
                    for floor in range(num_layers)]

        stacked_cell = StackedRNNCell(rnns, v_dropout)
        return stacked_cell
