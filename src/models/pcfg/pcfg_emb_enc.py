from typing import Tuple, List

import torch

from models.interfaces.encoder import EmbedAndEncode
from models.pcfg.base_pcfg import PCFGModule
from allennlp.nn.util import masked_max
import logging


class PCFGEmbedEncode(EmbedAndEncode):

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

        self.infer_net = None
        if z_dim > 0:
            self.infer_net = torch.nn.Linear(emb_enc.get_output_dim(), self.z_dim * 2)
        else:
            logging.getLogger(self.__class__.__name__).info(
                'Using Non-compound PCFG encoders, set z_dim > 0 if this is not expected.'
            )

        self._loss = None

    def forward(self, tokens: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        # mem: (b, n * (n - 1), hid)
        logPx, kl, mem, x_mask = self.encode(tokens)
        state = [mem]
        state_mask = self.get_state_mask(x_mask)

        self._loss = (-logPx + kl).mean()
        return state, state_mask

    def encode(self, x: torch.Tensor, x_hid: torch.Tensor = None):
        assert x is not None
        x_hid_, x_mask = self.embed_x(x)
        x_hid = x_hid or x_hid_

        # q_*: (batch, z_dim)
        if self.z_dim > 0:
            q_mean, q_logvar = self.run_inference(x_hid_, x_mask)
            kl = self.kl(q_mean, q_logvar).sum(1)
            z = self.reparameterized_sample(q_mean, q_logvar)
        else:
            kl = 0
            z = x_hid.new_zeros((x_hid.size()[0], 0))

        pcfg_params = self.pcfg.get_pcfg_params(z)
        logPx, mem = self.pcfg.inside(x, pcfg_params, x_hid)
        return logPx, kl, mem, x_mask

    @staticmethod
    def kl(mean, logvar):
        # mean, logvar: (batch, z_dim)
        result = -0.5 * (logvar - torch.pow(mean, 2) - torch.exp(logvar) + 1)
        return result

    def run_inference(self, state, state_mask):
        pooled = masked_max(state, state_mask.bool().unsqueeze(-1), dim=1)  # (b, d)
        q_mean, q_logvar = self.infer_net(pooled).split(self.z_dim, dim=1)
        return q_mean, q_logvar

    def embed_x(self, x):
        layered_state, state_mask = self.emb_enc(x)  # (b, N, d)
        state = layered_state[-1]
        return state, state_mask

    def reparameterized_sample(self, q_mean, q_logvar):
        z = q_mean  # use q_mean to approximate during evaluation
        if self.training:
            noise = q_mean.new_zeros(q_mean.size()).normal_(0, 1)
            z = q_mean + (.5 * q_logvar).exp() * noise  # z = u + sigma * noise
        return z

    def get_loss(self):
        return self._loss

    def get_state_mask(self, x_mask):
        b, n = x_mask.size()
        lengths = x_mask.long().sum(-1)
        length_chart_mask = x_mask.new_ones(b, n, n)
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
