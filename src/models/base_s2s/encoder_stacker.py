import torch.nn
from models.modules.variational_dropout import VariationalDropout
from ..interfaces.encoder import Encoder, StackEncoder
from allennlp.nn.util import sort_batch_by_length
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence


class EncoderStacker(StackEncoder):
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

    def __init__(self, encs: list[Encoder],
                 input_size: int | None = None,
                 output_size: int | None = None,
                 input_dropout: float = 0.,
                 ):
        super(EncoderStacker, self).__init__()

        self.layer_encs = torch.nn.ModuleList(encs)
        self.input_size = input_size or encs[0].get_input_dim()
        self.output_size = output_size
        self.input_dropout = VariationalDropout(input_dropout, on_the_fly=True)
        self._layered_output = None

    def forward(self, inputs: torch.Tensor, mask: None | torch.LongTensor):
        self._layered_output = None

        last_output = inputs
        layered_output = []
        for i, enc in enumerate(self.layer_encs):
            if i > 0:
                last_output = self.input_dropout(last_output)

            last_output = enc(last_output, mask)
            layered_output.append(last_output)

        self._layered_output = layered_output
        enc_output = layered_output[-1]
        return enc_output

    def get_layered_output(self) -> list[torch.Tensor]:
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


class ExtLSTM(Encoder):
    """
    A wrapper for nn.lstm supports both tensor and padded_seq protocols, used for StackedEncoder.
    """
    def forward(self, inputs, mask) -> torch.Tensor:
        if not self.use_packed_sequence:
            out, _ = self.lstm(inputs)
            return out

        # mask: (batch, length)
        # hidden: [(batch, hid), (batch, hid)], since its only for lstm
        (
            sorted_inputs,  # (batch, length, input_dim)
            sorted_sequence_lengths,  # (batch,), the descending lengths
            restoration_indices,  # (batch,), indices: sorted -> original
            sorting_indices,  # (batch,), indices: original -> sorted
        ) = sort_batch_by_length(inputs, mask.sum(-1))

        packed_input: PackedSequence = pack_padded_sequence(
            sorted_inputs,
            sorted_sequence_lengths.data.tolist(),
            batch_first=True,
        )
        packed_output, _ = self.lstm(packed_input)
        unpacked_tensor, _ = pad_packed_sequence(packed_output, batch_first=True)
        out = unpacked_tensor.index_select(0, restoration_indices)
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

