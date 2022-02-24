from typing import Optional, List
import torch.nn
from models.modules.variational_dropout import VariationalDropout
from ..interfaces.unified_rnn import EncoderRNNStack, EncoderRNN
from allennlp.nn.util import sort_batch_by_length
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence


class StackedEncoder(EncoderRNNStack):
    """
    Stacked Encoder over a token sequence. The encoder could be either based on RNN or Transformer.
    The Encoder must not be used with any initial hidden states
    because
    1) the user must know the internals of the hidden states in that case,
    2) only RNN models accept the initial hidden states, transformers do not have that.

    If it is a must, then refer to the models.base_s2s.rnn_lm.RNN_LM which is defined via the StackedRNNCells,
    But the stacked rnn cells do not support bidirectional forward by default.
    The allennlp.modules.seq2seq_encoders.PytorchSeq2SeqWrapper is also useful.
    """

    def __init__(self, encs,
                 input_size: Optional = None,
                 output_size: Optional = None,
                 input_dropout=0.,
                 ):
        super(StackedEncoder, self).__init__()

        self.layer_encs = torch.nn.ModuleList(encs)
        self.input_size = input_size or encs[0].get_input_dim()
        self.output_size = output_size
        self.input_dropout = VariationalDropout(input_dropout, on_the_fly=True)
        self._layered_output = None

    def forward(self,
                inputs: torch.Tensor,
                mask: Optional[torch.LongTensor],
                ):
        self._layered_output = None

        last_output = inputs
        layered_output = []
        for i, enc in enumerate(self.layer_encs):
            if i > 0:
                last_output = self.input_dropout(last_output)

            last_output = enc(last_output, mask, None)
            layered_output.append(last_output)

        self._layered_output = layered_output
        enc_output = layered_output[-1]
        return enc_output

    def get_last_layered_output(self) -> List[torch.Tensor]:
        return self._layered_output

    def get_layer_num(self):
        return len(self.layer_encs)

    def get_input_dim(self):
        return self.input_size

    def get_output_dim(self):
        if self.output_size is not None:
            out_dim = self.output_size * 2 if self.is_bidirectional() else self.output_size
        else:
            out_dim = self.layer_encs[-1].get_output_dim()
        return out_dim

    def is_bidirectional(self) -> bool:
        return self.layer_encs[-1].is_bidirectional()

    @classmethod
    def get_encoder(cls, p):
        """
        p.enc_dropout = 0.
        p.enc_out_dim = xxx # otherwise p.hidden_sz is used
        p.emb_sz = 300
        p.encoder = "lstm"  # lstm, transformer, bilstm, aug_lstm, aug_bilstm
        p.num_heads = 8     # heads for transformer when used
        """
        from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, AugmentedLstmSeq2SeqEncoder
        from allennlp.modules.seq2seq_encoders import StackedBidirectionalLstmSeq2SeqEncoder

        dropout = getattr(p, 'enc_dropout', getattr(p, 'dropout', 0.))
        hid_sz = getattr(p, 'enc_out_dim', p.hidden_sz)

        if p.encoder == "lstm":
            enc_cls = lambda floor: PytorchSeq2SeqWrapper(
                torch.nn.LSTM(p.emb_sz if floor == 0 else hid_sz, hid_sz, batch_first=True)
            )
        elif p.encoder == "transformer":
            from models.transformer.encoder import TransformerEncoder
            enc_cls = lambda floor: TransformerEncoder(
                input_dim=p.emb_sz if floor == 0 else hid_sz,
                hidden_dim=hid_sz,
                num_layers=1,
                num_heads=p.num_heads,
                feedforward_hidden_dim=hid_sz,
                feedforward_dropout=dropout,
                residual_dropout=dropout,
                attention_dropout=0.,
                use_positional_embedding=(floor == 0),
            )
        elif p.encoder == "bilstm":
            enc_cls = lambda floor: PytorchSeq2SeqWrapper(torch.nn.LSTM(
                p.emb_sz if floor == 0 else hid_sz * 2, hid_sz, bidirectional=True, batch_first=True,
            ))
        elif p.encoder == "torch_bilstm":
            enc_cls = lambda floor: ExtLSTM(torch.nn.LSTM(
                p.emb_sz if floor == 0 else hid_sz * 2, hid_sz, bidirectional=True, batch_first=True
            ))

        elif p.encoder == "aug_lstm":
            enc_cls = lambda floor: AugmentedLstmSeq2SeqEncoder(p.emb_sz if floor == 0 else hid_sz, hid_sz,
                                                                recurrent_dropout_probability=dropout,
                                                                use_highway=True, )
        elif p.encoder == "aug_bilstm":
            enc_cls = lambda floor: StackedBidirectionalLstmSeq2SeqEncoder(
                p.emb_sz if floor == 0 else hid_sz, hid_sz, num_layers=1,
                recurrent_dropout_probability=dropout,
                use_highway=True,
            )
        else:
            raise NotImplementedError

        encoder = StackedEncoder([enc_cls(floor) for floor in range(p.num_enc_layers)],
                                 input_size=p.emb_sz,
                                 output_size=hid_sz,
                                 input_dropout=dropout)
        return encoder


class ExtLSTM(EncoderRNN):
    def forward(self, inputs, mask, hidden) -> torch.Tensor:
        if hidden is not None:
            raise NotImplementedError('only the none state is supported now.')

        if not self.use_packed_sequence:
            out, _ = self.lstm(inputs, hidden)
            return out

        # mask: (batch, length)
        # hidden: [(batch, hid), (batch, hid)], since its only for lstm
        (
            sorted_inputs,  # (batch, length, input_dim)
            sorted_sequence_lengths,  # (batch,), the descending lengths
            restoration_indices,  # (batch,), indices: sorted -> original
            sorting_indices,  # (batch,), indices: original -> sorted
        ) = sort_batch_by_length(inputs, mask.sum(-1))

        packed_sequence_input: PackedSequence = pack_padded_sequence(
            sorted_inputs,
            sorted_sequence_lengths.data.tolist(),
            batch_first=True,
        )
        packed_sequence_output, _ = self.lstm(packed_sequence_input, None)
        unpacked_sequence_tensor, _ = pad_packed_sequence(packed_sequence_output,
                                                          batch_first=True)
        out = unpacked_sequence_tensor.index_select(0, restoration_indices)
        return out

    def __init__(self, lstm: torch.nn.LSTM, use_packed_sequence_protocol: bool = True):
        super().__init__()
        self.lstm = lstm
        self.use_packed_sequence = use_packed_sequence_protocol

    def is_bidirectional(self) -> bool:
        return self.lstm.bidirectional

    def get_input_dim(self) -> int:
        return self.lstm.input_size

    def get_output_dim(self) -> int:
        num_directions = 2 if self.is_bidirectional() else 1
        return self.lstm.hidden_size * num_directions

