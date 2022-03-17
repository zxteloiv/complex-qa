import torch
from typing import Optional, List, Any
from ..interfaces.unified_rnn import UnifiedRNN
from models.interfaces.encoder import Encoder
from utils.seq_collector import SeqCollector
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch.nn.functional import pad
from allennlp.nn.util import sort_batch_by_length


class CellEncoder(Encoder):
    def forward(self, inputs, mask, hidden) -> torch.Tensor:
        if not self.use_packed_seq:
            return self.forward_tensor_seq(inputs, mask, hidden)

        else:
            if hidden is not None:
                raise NotImplementedError("the packed sequence protocol only allows None for initial states")

            (
                sorted_inputs,  # (batch, length, input_dim)
                sorted_sequence_lengths,  # (batch,), the descending lengths
                restoration_indices,  # (batch,), indices: sorted -> original
                sorting_indices,  # (batch,), indices: original -> sorted
            ) = sort_batch_by_length(inputs, mask.sum(-1))

            pseq = pack_padded_sequence(sorted_inputs, sorted_sequence_lengths.tolist(), batch_first=True)
            sorted_out = self.forward_packed_seq(pseq, None)
            out = sorted_out.index_select(0, restoration_indices)
            return out

    def forward_tensor_seq(self, inputs, mask, hidden) -> torch.Tensor:
        """
        :param seq: (batch, seq_len, input_dim)
        :param mask: (batch, seq_len), not required for non-bidirectional cells
        :param hx: Any
        :return: (batch, seq_len, hid [*2])
        """
        forward_out = self._forward_seq(inputs, hidden)
        if not self.is_bidirectional():
            return forward_out

        # when running backwards, we make two assumptions
        # 1) the init hx is the same as the forward pass, because the hx is supposed to enrich the sequence embedding
        # 2) the mask will not get processed explicitly due to the different behavior within the batch, in practice,
        #    the hx will be always 0 if the padding idx is set properly in the nn.Embedding objects
        back_out = self._forward_seq(inputs, hidden, is_reversed=True)
        # forward_out, back_out: (batch, len, hid)
        # bi_out: (batch, len, hid * 2)
        bi_out = torch.cat([forward_out, back_out], dim=-1)
        return bi_out

    def _forward_seq(self, seq: torch.Tensor, hx: Any = None, is_reversed: bool = False):
        if is_reversed:
            cell = self.back_cell
            steps = list(reversed(range(seq.size()[1])))
        else:
            cell = self.cell
            steps = list(range(seq.size()[1]))

        mem = SeqCollector()
        for step in steps:
            step_input = seq[:, step]
            hx, out = cell(step_input, hx)  # out: (batch, hid)
            mem(out=out)

        def _proper_reverse(x): return list(reversed(x)) if is_reversed else x

        # enc_output: (batch, len, hid)
        return torch.stack(_proper_reverse(mem['out']), dim=1)

    def forward_packed_seq(self, seq: PackedSequence, hx: Any = None):
        data, batch_sizes = seq.data, seq.batch_sizes
        ltr_output = self._forward_pseq_ltr(data, batch_sizes)
        if not self.is_bidirectional():
            return ltr_output

        rtl_output = self._forward_pseq_rtl(data, batch_sizes)
        output = torch.cat([ltr_output, rtl_output], dim=-1)
        return output

    def _forward_pseq_ltr(self, data: torch.Tensor, batch_sizes: torch.Tensor):
        outputs = []
        input_offset = 0
        last_batch_size = batch_sizes[0]
        max_batch_size = batch_sizes[0].item()

        hx, _ = self.cell.forward(data[:max_batch_size], None)
        flat_hx = isinstance(hx, torch.Tensor)
        # print('is_flat_hx:', flat_hx)
        if flat_hx:
            hx = (hx,)
        initial_hidden = tuple(torch.zeros_like(h) for h in hx)
        # print('init hiddens:', *[h.size() for h in initial_hidden])

        hidden = tuple(h[:max_batch_size] for h in initial_hidden)

        def is_t(x): return isinstance(x, torch.Tensor)

        for batch_size in batch_sizes:
            step_input = data[input_offset:input_offset + batch_size]
            input_offset += batch_size
            dec = last_batch_size - batch_size

            if dec > 0:
                hidden = tuple(h[:-dec] if is_t(h) else h for h in hidden)

            last_batch_size = batch_size
            # print('cell input:', step_input.size(), *[h.size() for h in hidden])
            hidden, o = self.cell(step_input, hidden[0] if flat_hx else hidden)
            if flat_hx:
                hidden = (hidden,)

            # print('cell output:', *[h.size() for h in hidden])
            outputs.append(o)

        output = torch.stack([
            pad(o, [0, 0, 0, max_batch_size - o.size()[0]])
            for o in outputs
        ], dim=1)
        return output

    def _forward_pseq_rtl(self, data: torch.Tensor, batch_sizes: torch.Tensor):
        max_batch_size = batch_sizes[0]

        # probe for the hidden state sizes
        hx, _ = self.back_cell.forward(data[:max_batch_size], None)
        flat_hx = isinstance(hx, torch.Tensor)
        if flat_hx:
            hx = (hx,)
        initial_hidden = tuple(torch.zeros_like(h) for h in hx)
        hidden = tuple(h[:batch_sizes[-1]] for h in initial_hidden)

        outputs, input_offset, last_batch_size = [], data.size()[0], batch_sizes[-1]
        for batch_size in reversed(batch_sizes):
            if batch_size > last_batch_size:
                hidden = tuple(
                    torch.cat((h, ih[last_batch_size:batch_size]), 0)
                    for h, ih in zip(hidden, initial_hidden)
                )
            last_batch_size = batch_size
            step_input = data[input_offset - batch_size:input_offset]
            input_offset -= batch_size
            hidden, o = self.back_cell(step_input, hidden[0] if flat_hx else hidden)
            if flat_hx:
                hidden = (hidden,)
            outputs.append(o)

        outputs.reverse()
        output = torch.stack([
            pad(o, [0, 0, 0, max_batch_size - o.size()[0]])
            for o in outputs
        ], dim=1)
        return output

    def is_bidirectional(self) -> bool:
        return self.back_cell is not None

    def get_input_dim(self) -> int:
        return self.cell.get_input_dim()

    def get_output_dim(self) -> int:
        return self.cell.get_output_dim() * 2 if self.is_bidirectional() else self.cell.get_output_dim()

    def __init__(self,
                 cell: UnifiedRNN,
                 back_cell: Optional[UnifiedRNN] = None,
                 use_packed_sequence_protocol: bool = False,
                 ):
        super().__init__()
        self.cell = cell
        self.back_cell = back_cell
        self.use_packed_seq = use_packed_sequence_protocol


if __name__ == '__main__':
    x = torch.tensor(
        [[16,  3, 18,  9,  5, 22, 17,  3, 21,  1],
         [27, 21,  1, 17,  7, 17,  6, 17,  0,  0],
         [15,  7, 31,  7, 20,  0,  0,  0,  0,  0],
         [17, 20, 23, 28, 15,  0,  0,  0,  0,  0],
         [24,  9, 28,  0,  0,  0,  0,  0,  0,  0],
         [ 4, 17, 27,  1, 24, 31, 19, 10, 23,  0],
         [23, 20, 21, 27, 27,  2, 16,  3,  4, 23],
         [17, 21, 19,  0,  0,  0,  0,  0,  0,  0]], dtype=torch.int32,
    )
    mask = (x > 0).long()
    embedder = torch.nn.Embedding(32, 100, padding_idx=0)
    print(embedder.weight[0, :4])
    emb = embedder(x)
    print(emb[:, :, 0])
    print('-' * 50)

    from models.modules.torch_rnn_wrapper import TorchRNNWrapper as RNNWrapper
    enc = CellEncoder(cell=RNNWrapper(torch.nn.GRUCell(100, 128)),
                      back_cell=RNNWrapper(torch.nn.GRUCell(100, 128)),
                      use_packed_sequence_protocol=True)
    o = enc(emb, mask, None)
    print(o.size())
    print(o[:, :, 0])
    print('-' * 50)
    from models.onlstm.onlstm import ONLSTMCell
    onenc = CellEncoder(cell=ONLSTMCell(100, 128, 8),
                        back_cell=ONLSTMCell(100, 128, 8),
                        use_packed_sequence_protocol=True)
    o = onenc(emb, mask, None)
    print(o.size())
    print(o[:, :, 0])

