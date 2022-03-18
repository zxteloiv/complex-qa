import torch
from .diora_encoder import DioraEncoder
from ..base_s2s.seq_embed_encode import SeqEmbedEncoder
from ..interfaces.encoder import EncoderStack
from utils.nn import prepare_input_mask, seq_cross_ent
from .hard_diora import DioraTopk


class SeqEmbedAndDiora(SeqEmbedEncoder):
    def __init__(self,
                 source_embedding: torch.nn.Embedding,
                 encoder: EncoderStack,
                 padding_index: int = 0,
                 enc_dropout: float = 0.,
                 use_diora_loss: bool = False
                 ):
        super().__init__(source_embedding, encoder, padding_index, enc_dropout)
        if not isinstance(self._encoder, DioraEncoder):
            raise TypeError("Diora2Seq instances must be assembled with"
                            "DioraEncoder instead of other StackedEncoder."
                            "Because it is forbidden to stack Diora with other type of encoders.")

        # use the settings similar to the original implementation, fixed embedding and train the projections;
        # there's already a leaf transformation within diora, so we only project the embeddings for the loss
        # src_vocab_size = self.vocab.get_vocab_size(self._source_namespace)
        _, emb_sz = self._src_embedding.weight.size()
        self.reconstruct = torch.nn.Linear(self._encoder.diora.size, emb_sz)
        # as shown in the de-attentions setup, the embedding is not kept fixed
        # self._src_embedding.weight.requires_grad = False
        self.use_diora_loss = use_diora_loss
        self._loss = 0

    def _get_reconstruct_loss(self, source, mask):
        self._encoder: DioraEncoder
        diora = self._encoder.diora
        batch_sz, length = mask.size()
        # cells: (batch, length, hidden)
        cells = diora.chart['outside_h'][:, :length]

        # proj_cells: (batch, length, emb_sz)
        proj_cells = self.reconstruct(cells)
        # emb_weight: (num_toks, emb_sz)
        emb_weight = self._src_embedding.weight
        # logits: (batch, length, num_toks)
        logits = torch.einsum('ble,ne->bln', proj_cells, emb_weight)

        loss = seq_cross_ent(logits, source, mask, average="token")
        return loss

    def _get_tree_loss(self, mask: torch.Tensor, margin=1):
        self._encoder: DioraEncoder
        # root_idx: (batch,)
        root_idx = self._encoder.get_root_cell_idx(mask)
        batch_sz = mask.size()[0]
        diora = self._encoder.diora
        batch_idx = torch.arange(batch_sz, device=mask.device)
        gold_score = diora.chart['inside_s'][batch_idx, root_idx].contiguous().view(batch_sz, 1)
        pred_score = diora.charts[1]['inside_s'][batch_idx, root_idx].contiguous().view(batch_sz, 1)

        # just follows the original definition
        tr_loss = torch.clamp(pred_score + margin - gold_score, min=0)
        # returned tr_loss: (,) <- (batch, 1)
        return tr_loss.mean()

    def get_loss(self):
        return self._loss

    def forward(self, tokens: torch.LongTensor):
        source, source_mask = prepare_input_mask(tokens, self._padding_index)
        source_embedding = self.embed(tokens)
        layered_hidden = self.encode(source_embedding, source_mask)
        state_mask = self._encoder.output_mask

        self._loss = 0
        if self.use_diora_loss:
            src, src_mask = prepare_input_mask(tokens, self._padding_index)
            reconstruct_loss = self._get_reconstruct_loss(src, src_mask)

            loss = reconstruct_loss

            self._encoder: DioraEncoder
            if isinstance(self._encoder.diora, DioraTopk):
                tr_loss = self._get_tree_loss(src_mask, margin=1)
                loss = loss + tr_loss

            self._loss = loss

        return layered_hidden, state_mask

    # @classmethod
    # def from_param_and_vocab(cls, p, vocab):
    #     model = super().from_param_and_vocab(p, vocab)
    #     model.use_diora_loss = getattr(p, 'diora_loss_enabled', False)
    #     return model
