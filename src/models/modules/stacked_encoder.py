from typing import Optional
import torch.nn

class StackedEncoder(torch.nn.Module):
    def __init__(self, encs,
                 input_size: Optional = None,
                 output_size: Optional = None,
                 input_dropout=0.,
                 output_every_layer=True,
                 ):
        super(StackedEncoder, self).__init__()

        self.layer_encs = torch.nn.ModuleList(encs)
        self.input_size = input_size or encs[0].get_input_dim()
        self.output_size = output_size or encs[-1].get_output_dim()
        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.output_every_layer = output_every_layer

    def forward(self,
                inputs: torch.Tensor,
                mask: Optional[torch.LongTensor],
                ):
        last_output = inputs
        layered_output = []
        for i, enc in enumerate(self.layer_encs):
            if i > 0:
                last_output = self.input_dropout(last_output)

            last_output = enc(last_output, mask)
            layered_output.append(last_output)

        enc_output = layered_output[-1]

        if self.output_every_layer:
            return enc_output, layered_output
        else:
            return enc_output

    def get_layer_num(self):
        return len(self.layer_encs)

    def get_input_dim(self):
        return self.input_size

    def get_output_dim(self):
        return self.output_size * 2 if self.is_bidirectional() else self.output_size

    def is_bidirectional(self) -> bool:
        return self.layer_encs[-1].is_bidirectional()

    @classmethod
    def get_encoder(cls, p):
        from models.transformer.encoder import TransformerEncoder
        from allennlp.modules.seq2seq_encoders import LstmSeq2SeqEncoder

        inp_sz_fn = lambda floor: p.emb_sz if floor == 0 else p.hidden_sz
        if p.encoder == "lstm":
            enc_cls = lambda floor: LstmSeq2SeqEncoder(inp_sz_fn(floor), p.hidden_sz, bidirectional=False)
        elif p.encoder == "transformer":
            enc_cls = lambda floor: TransformerEncoder(
                inp_sz_fn(floor), p.hidden_sz, 1, p.num_heads, p.hidden_sz, p.dropout, p.dropout, 0., (floor == 0),
            )
        elif p.encoder == "bilstm":
            enc_cls = lambda floor: LstmSeq2SeqEncoder(inp_sz_fn(floor), p.hidden_sz, bidirectional=True)
        else:
            raise NotImplementedError

        encoder = StackedEncoder([enc_cls(floor) for floor in range(p.num_enc_layers)], input_dropout=p.dropout)
        return encoder

