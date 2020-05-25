import torch
from torch import nn
from allennlp.nn.util import get_final_encoder_states


class WordCharEmbedding(nn.Module):
    """
    Concatenate the word embedding and char embedding of the appropriate word.
    """
    def __init__(self,
                 word_emb: nn.Module,
                 char_emb: nn.Module,
                 char_encoder: nn.Module,
                 ):
        super().__init__()
        self.word_emb = word_emb
        self.char_emb = char_emb
        self.char_encoder = char_encoder

    def forward(self,
                word_ids: torch.LongTensor,
                char_ids: torch.LongTensor,
                char_mask: torch.LongTensor,
                ) -> torch.Tensor:
        """
        Run forward to get concatenated embeddings of every word
        :param word_ids: (batch, max_word_count)
        :param char_ids: (batch, max_word_count, max_char_count)
        :param char_mask: (batch, max_word_count, max_char_count)
        :return: the concatenated word embedding: (batch, L, word_emb + char_emb)
        """

        # word_emb: (batch, L, word_emb)
        word_emb = self.word_emb(word_ids)

        # char_emb: (batch, L, C, char_emb)
        char_emb = self.char_emb(char_ids)
        orig_size = char_emb.size() # (batch, L, C, char_emb)

        # char_emb_rs: (batch * L, C, char_emb)
        # char_mask_rs: (batch * L, C)
        char_emb_rs = char_emb.reshape(-1, *orig_size[-2:])
        char_mask_rs = char_mask.reshape(-1, orig_size[-2])

        # hid: (batch * L, C, hid_emb)
        hid = self.char_encoder(char_emb_rs, char_mask_rs)

        # final_state: (batch * L, hid_emb)
        # final_char_state: (batch, L, hid_emb)
        final_state = get_final_encoder_states(hid, char_mask_rs, bidirectional=False)
        final_char_state = final_state.reshape(orig_size[:2], -1)

        # concat_word_emb: (batch, L, word_emb + hid_emb)
        concat_word_emb = torch.cat([word_emb, final_char_state], dim=-1)
        return concat_word_emb

