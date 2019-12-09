from typing import Tuple
import torch
import math
from torch import nn
from allennlp.nn.util import masked_softmax

class Re2Block(nn.Module):
    def __init__(self,
                 a_encoder,
                 b_encoder,
                 a_fusion,
                 b_fusion,
                 alignment,
                 dropout: float = 0.2,
                 ):
        super().__init__()
        self.a_enc = a_encoder
        self.b_enc = b_encoder
        self.a_fusion = a_fusion
        self.b_fusion = b_fusion
        self.alignment = alignment
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                a: torch.Tensor,
                b: torch.Tensor,
                mask_a: torch.LongTensor,
                mask_b: torch.LongTensor):
        raw_a, raw_b = a, b
        a = self.a_enc(a, mask_a)
        b = self.b_enc(b, mask_b)

        a = torch.cat([raw_a, a], dim=-1)
        b = torch.cat([raw_b, b], dim=-1)

        align_a, align_b = self.alignment(a, b, mask_a, mask_b)
        a = self.a_fusion(a, align_a)
        b = self.b_fusion(b, align_b)
        return a, b

class Re2Dense(nn.Module):
    def __init__(self, in_size, out_size,
                 activation: nn.Module = None,
                 use_bias: bool = True,
                 use_weight_norm: bool = True,
                 ):
        super().__init__()
        dense = nn.Linear(in_size, out_size, bias=use_bias)
        # try to use the same initializers as the RE2 official TF code,
        # according to the original `gain = sqrt(2.) if activation is relu else 1`,
        # nonliearity is set to "linear" except "relu" for relu
        if any(cls_name in str(activation.__class__).lower() for cls_name in ('relu', 'gelu')):
            nonlinear = "relu"
        else:
            nonlinear = "linear"
        nn.init.kaiming_normal_(dense.weight, nonlinearity=nonlinear)
        nn.init.zeros_(dense.bias)

        self.dense = dense
        if use_weight_norm:
            self.dense = nn.utils.weight_norm(self.dense, 'weight')

        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation is None:
            return self.dense(x)
        else:
            return self.activation(self.dense(x))

class Re2Encoder(nn.Module):
    def __init__(self, stacked_conv, activation = None, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.stacked_conv = stacked_conv
        self.activation = activation

    def forward(self, x: torch.Tensor, mask: torch.LongTensor) -> torch.Tensor:
        """
        :param x: (batch, len, embedding)
        :param mask: (batch, len)
        :return: (batch, len, embedding)
        """
        inp = x * mask.unsqueeze(-1).float()
        inp = inp.transpose(1, 2)   # convert to (batch, embedding, len) for conv1d: (N, C, L)
        for conv1d in self.stacked_conv:
            inp = conv1d(inp)   # output channel should be the same as the input
            if self.activation:
                inp = self.activation(inp)
            inp = self.dropout(inp)

        rtn = inp.transpose(1, 2)   # back to (batch, len, embedding)
        return rtn

    @staticmethod
    def from_re2_default(num_stacked_layers: int,
                         inp_sz: int,
                         hidden: int,
                         kernel_sz: int,
                         dropout: float):
        """
        :param num_stacked_layers: the encoder is actually stacked conv layers
        :param inp_sz: input size, which is embedding size or augmented residual size
        :param hidden:
        :param kernel_sz:
        :param dropout:
        :return:
        """
        stacked_conv = nn.ModuleList([
            nn.Conv1d(inp_sz if i == 0 else hidden, hidden, kernel_sz, padding=kernel_sz//2)
            for i in range(num_stacked_layers)
        ])
        return Re2Encoder(stacked_conv, activation=nn.ReLU(), dropout=dropout)

class Re2Fusion(nn.Module):
    def __init__(self,
                 inp_sz: int,
                 hid_sz: int,
                 use_full_mode: bool = True,
                 dropout: float = 0.2):
        """
        :param inp_sz: input size of x, and alignment
        :param hid_sz: the output size of fusion
        :param use_full_mode:
        :param dropout:
        """
        super().__init__()
        self.use_full_mode = use_full_mode
        self.feat_mappings = nn.ModuleList([
            Re2Dense(inp_sz * 2, hid_sz, nn.ReLU())
            for _ in range(3)   # 3 heuristic features in total
        ])
        self.projection = Re2Dense(hid_sz * 3, hid_sz, nn.ReLU())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, align):
        if self.use_full_mode:
            features = [
                torch.cat([x, align], dim=-1),
                torch.cat([x, x - align], dim=-1),
                torch.cat([x, x * align], dim=-1)
            ]
            mapped_features = [f(x) for f, x in zip(self.feat_mappings, features)]
            concated_features = self.dropout(torch.cat(mapped_features, dim=-1))
            rtn = self.projection(concated_features)

        else:
            features = torch.cat([x, align], dim=-1)
            rtn = self.feat_mappings[0](features)

        return rtn

class Re2Alignment(nn.Module):
    def __init__(self, inp_sz: int, hid_sz: int, mode: str):
        super().__init__()
        tau = nn.Parameter(torch.randn(size=(), ), requires_grad=True)
        nn.init.constant_(tau, val=math.sqrt(1 / hid_sz))
        self.tau = tau

        self.mode = mode
        if mode == "linear":
            self.a_linear = Re2Dense(inp_sz, hid_sz, activation=nn.ReLU())
            self.b_linear = Re2Dense(inp_sz, hid_sz, activation=nn.ReLU())
        elif mode == "bilinear":
            from allennlp.modules.matrix_attention import BilinearMatrixAttention
            self.a_linear = Re2Dense(inp_sz, hid_sz, activation=nn.ReLU())
            self.b_linear = Re2Dense(inp_sz, hid_sz, activation=nn.ReLU())
            self.bilinear = BilinearMatrixAttention(inp_sz, inp_sz, activation=nn.ReLU(), use_input_biases=True)

    def forward(self, a: torch.Tensor, b: torch.Tensor,
                mask_a: torch.LongTensor, mask_b: torch.LongTensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param a: (batch, len_a, inp_sz)
        :param b: (batch, len_b, inp_sz)
        :param mask_a: (batch, len_a)
        :param mask_b: (batch, len_b)
        :return:
        """
        # attn_strength: (batch, len_a, len_b)
        if self.mode == "linear":
            attn_strength = self._linear_attn(a, b)
        elif self.mode == "bilinear":
            attn_strength = self._bilinear_attn(a, b)
        else:
            attn_strength = self._identity_attn(a, b)

        # weights (a and b): (batch, len_a, len_b)
        weight_a = masked_softmax(attn_strength, mask_a.unsqueeze(2), dim=1)
        weight_b = masked_softmax(attn_strength, mask_b.unsqueeze(1), dim=2)

        # align_a: (batch, len_a, inp_sz)
        # align_b: (batch, len_b, inp_sz)
        align_a = weight_b.matmul(b)
        align_b = weight_a.transpose(1, 2).matmul(a)
        return align_a, align_b

    def _identity_attn(self, a: torch.Tensor, b: torch.Tensor):
        return torch.bmm(a, b.transpose(1, 2)) * self.tau

    def _linear_attn(self, a: torch.Tensor, b: torch.Tensor):
        a = self.a_linear(a)
        b = self.b_linear(b)
        return self._identity_attn(a, b)

    def _bilinear_attn(self, a, b):
        a = self.a_linear(a)
        b = self.b_linear(b)
        return self.bilinear(a, b) * self.tau

class Re2Pooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        """
        :param x: (batch, len, embedding)
        :param mask: (batch, len)
        :return: (batch, embedding)
        """
        return ((mask.unsqueeze(-1) + 1e-45).log() + x).max(dim=1)[0]

class Re2Conn(nn.Module):
    def __init__(self, mode: str, emb_sz: int, hid_sz: int):
        """
        :param mode: Chosen from "none", "residual", and "aug"
        """
        super().__init__()
        self.mode = mode.lower()
        self.res_proj = None
        if mode == "residual":  # only needed in residual mode
            self.res_proj = Re2Dense(hid_sz, emb_sz)
        self.hid_sz = hid_sz
        self.emb_sz = emb_sz

    def forward(self, x, res, layer_depth: int):
        if self.mode == "none":
            return x
        elif self.mode == "residual":
            return (self.res_proj(x) + res) * math.sqrt(0.5)

        # "aug" mode
        # at layer 0, no residual connection could appear
        # at layer 1, residual connection is the embedding itself
        # for layer >1, residual connection will be added before embedding is concatenated
        assert layer_depth >= 1
        if layer_depth == 1:
            raw_emb = res
        else:
            raw_emb = res[:, :, :-self.hid_sz]
            x = (x + res[:, :, -self.hid_sz:]) * math.sqrt(0.5)

        return torch.cat([raw_emb, x], dim=-1)

    def get_output_size(self):
        if self.mode == "none":
            return self.hid_sz  # same size with the output of fusion
        elif self.mode == "residual":
            return self.emb_sz  # same size with res connection, starting from embedding
        elif self.mode == "aug":
            return self.emb_sz + self.hid_sz

class Re2Prediction(nn.Module):
    def __init__(self, mode: str, inp_sz: int, hid_sz: int, num_classes: int, dropout: float = .2):
        super().__init__()
        self.mode = mode.lower()
        if mode in ("full", "symmetric"):
            feat_sz = inp_sz * 4
        else:
            feat_sz = inp_sz * 2

        self.dense1 = Re2Dense(feat_sz, hid_sz, activation=nn.ReLU())
        self.dense2 = Re2Dense(hid_sz, num_classes, activation=None)
        self.dropout = nn.Dropout(dropout)

    def forward(self, a, b):
        feature = self._get_features(a, b)
        feature = self.dropout(feature)
        feature = self.dense1(feature)
        feature = self.dropout(feature)
        prediction = self.dense2(feature)
        return prediction

    def _get_features(self, a, b):
        if self.mode == "full":
            return torch.cat([a, b, a * b, a - b], dim=-1)
        elif self.mode == "symmetric":
            return torch.cat([a, b, a * b, torch.abs(a - b)], dim=-1)
        else:
            return torch.cat([a, b], dim=-1)




