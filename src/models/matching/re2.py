# Reference to paper
# @inproceedings{yang2019simple,
#   title={Simple and Effective Text Matching with Richer Alignment Features},
#   author={Yang, Runqi and Zhang, Jianhai and Gao, Xing and Ji, Feng and Chen, Haiqing},
#   booktitle={Association for Computational Linguistics (ACL)},
#   year={2019}
# }
# Original code in tensorflow at https://github.com/alibaba-edu/simple-effective-text-matching
#

import torch
from torch import nn
from .re2_modules import Re2Block, Re2Prediction, Re2Pooling, Re2Conn
from .re2_modules import Re2Encoder, Re2Alignment, Re2Fusion

class RE2(nn.Module):
    def __init__(self,
                 a_embedding,
                 b_embedding,
                 blocks: nn.ModuleList,
                 a_pooling,
                 b_pooling,
                 connection,
                 prediction,
                 padding_val_a,
                 padding_val_b,
                 ):
        super().__init__()
        self.a_emb = a_embedding
        self.b_emb = b_embedding
        self.blocks = blocks
        self.a_pooling = a_pooling
        self.b_pooling = b_pooling
        self.prediction = prediction
        self.connection = connection
        self.padding_val_a = padding_val_a
        self.padding_val_b = padding_val_b

    def forward(self, sent_a: torch.LongTensor, sent_b: torch.LongTensor) -> torch.Tensor:
        """
        :param sent_a: (batch, max_a_len)
        :param sent_b: (batch, max_b_len)
        :return: (batch, num_classes)
        """
        a = self.a_emb(sent_a)
        b = self.b_emb(sent_b)
        mask_a = (sent_a != self.padding_val_a).long()
        mask_b = (sent_b != self.padding_val_b).long()
        return self.forward_embs(a, b, mask_a, mask_b)

    def forward_embs(self, a, b, mask_a, mask_b):

        res_a, res_b = a, b
        for i, block in enumerate(self.blocks):
            if i > 0:
                a = self.connection(a, res_a, i)
                b = self.connection(b, res_b, i)
                res_a, res_b = a, b

            a, b = block(a, b, mask_a, mask_b)

        a = self.a_pooling(a, mask_a)
        b = self.b_pooling(b, mask_b)
        logits = self.prediction(a, b)
        return logits

    @staticmethod
    def get_model(emb_sz: int,
                  num_tokens_a: int,
                  num_tokens_b: int,
                  hid_sz: int,
                  enc_kernel_sz: int,
                  num_classes: int,
                  num_stacked_blocks: int,
                  num_encoder_layers: int,
                  dropout: float,
                  fusion_mode: str, # simple or full
                  alignment_mode: str,  # identity or linear
                  connection_mode: str, # none or residual or "aug"
                  prediction_mode: str, # simple, full, or symmetric
                  use_shared_embedding: bool = False,
                  padding_val_a: int = 0,
                  padding_val_b: int = 0,
                  ) -> 'RE2':
        embedding_a = nn.Embedding(num_tokens_a, emb_sz)
        embedding_b = embedding_a if use_shared_embedding else nn.Embedding(num_tokens_b, emb_sz)

        conn: Re2Conn = Re2Conn(connection_mode, emb_sz, hid_sz)
        conn_out_sz = conn.get_output_size()

        # the input to predict is exactly the output of fusion, with the hidden size
        pred = Re2Prediction(prediction_mode, inp_sz=hid_sz, hid_sz=hid_sz, num_classes=num_classes, dropout=dropout)
        pooling = Re2Pooling()

        enc_inp_sz = lambda i: emb_sz if i == 0 else conn_out_sz
        blocks = nn.ModuleList([
            Re2Block(
                Re2Encoder.from_re2_default(num_encoder_layers, enc_inp_sz(i), hid_sz, enc_kernel_sz, dropout),
                Re2Encoder.from_re2_default(num_encoder_layers, enc_inp_sz(i), hid_sz, enc_kernel_sz, dropout),
                Re2Fusion(hid_sz + enc_inp_sz(i), hid_sz, fusion_mode == "full", dropout),
                Re2Fusion(hid_sz + enc_inp_sz(i), hid_sz, fusion_mode == "full", dropout),
                Re2Alignment(hid_sz + enc_inp_sz(i), hid_sz, alignment_mode),
                dropout=dropout,
            )
            for i in range(num_stacked_blocks)
        ])

        model = RE2(embedding_a, embedding_b, blocks, pooling, pooling, conn, pred, padding_val_a, padding_val_b)
        return model
