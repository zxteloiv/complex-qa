from typing import Tuple, List

import torch

from models.interfaces.encoder import EmbedAndEncode
from models.pcfg.base_pcfg import PCFGModule
from allennlp.nn.util import masked_max


class CompoundPCFGEmbedEncode(EmbedAndEncode):

    def __init__(self,
                 pcfg: PCFGModule,
                 emb_enc: EmbedAndEncode,
                 z_dim: int,
                 padding: int = 0,
                 ):
        super().__init__()
        self.pcfg = pcfg
        self.emb_enc = emb_enc
        self.z_dim = z_dim
        self.padding = padding
        self.infer_net = torch.nn.Linear(emb_enc.get_output_dim(), self.z_dim * 2)
        self._loss = None

    def forward(self, tokens: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        # mem: (b, n * (n - 1), hid)
        logPx, kl, mem = self.encode(tokens)
        state = [mem]
        state_mask = self.get_state_mask(tokens)

        self._loss = (-logPx + kl).mean()
        return state, state_mask

    def encode(self, x: torch.Tensor, x_hid: torch.Tensor = None):
        assert x is not None
        # q_*: (batch, z_dim)
        x_hid_, q_mean, q_logvar = self.run_inference(x)
        x_hid = x_hid or x_hid_

        kl = self.kl(q_mean, q_logvar).sum(1)
        z = self.reparameterized_sample(q_mean, q_logvar)
        pcfg_params = self.pcfg.get_pcfg_params(z)

        logPx, mem = self.pcfg.inside(x, pcfg_params, x_hid)
        return logPx, kl, mem

    @staticmethod
    def kl(mean, logvar):
        # mean, logvar: (batch, z_dim)
        result = -0.5 * (logvar - torch.pow(mean, 2) - torch.exp(logvar) + 1)
        return result

    def run_inference(self, x):
        layered_state, state_mask = self.emb_enc(x)  # (b, N, d)
        state = layered_state[-1]
        pooled = masked_max(state, state_mask.bool().unsqueeze(-1), dim=1)  # (b, d)
        q_mean, q_logvar = self.infer_net(pooled).split(self.z_dim, dim=1)
        return state, q_mean, q_logvar

    def reparameterized_sample(self, q_mean, q_logvar):
        z = q_mean  # use q_mean to approximate during evaluation
        if self.training:
            noise = q_mean.new_zeros(q_mean.size()).normal_(0, 1)
            z = q_mean + (.5 * q_logvar).exp() * noise  # z = u + sigma * noise
        return z

    def get_loss(self):
        return self._loss

    def get_state_mask(self, tokens: torch.Tensor):
        b, n = tokens.size()
        lengths = (tokens != self.padding).long().sum(-1)
        length_chart_mask = tokens.new_ones(b, n, n)
        # iterate over batch instead of instantiating a length mask lookup
        for length, chart in zip(lengths, length_chart_mask):
            chart.tril_(length - n)
        length_chart_mask = length_chart_mask.rot90(-1, (1, 2))
        length_chart_mask = length_chart_mask[:, 1:, :].reshape(b, n * (n - 1))
        return length_chart_mask

    def is_bidirectional(self):
        return False

    def get_output_dim(self) -> int:
        return self.pcfg.get_encoder_output_size()
