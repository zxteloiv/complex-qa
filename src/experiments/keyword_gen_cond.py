from typing import List, Generator, Tuple, Mapping
import os.path
import config
import torch
from torch.nn.utils.rnn import pad_sequence
from allennlp.modules import Embedding
from data_adapter.ns_vocabulary import NSVocabulary
from data_adapter.general_datasets.tabular_dataset import TabSepFileDataset
from models.transformer.keyword_conditioned_seq2seq import KeywordConditionedTransformer, Decoder, DecoderLayer
from models.transformer.encoder import TransformerEncoder
from models.modules.mixture_softmax import MoSProjection
from data_adapter.ns_vocabulary import START_SYMBOL, END_SYMBOL, PADDING_TOKEN
from training.trial_bot.trial_registry import Registry
from data_adapter.translator import Translator
import logging

@Registry.hparamset()
def weibo_keyword_gen():
    hparams = config.common_settings()
    hparams.emb_sz = 300
    hparams.batch_sz = 50
    hparams.num_layers = 2
    hparams.num_heads = 8
    hparams.max_decoding_len = 30
    hparams.ADAM_LR = 1e-5
    hparams.TRAINING_LIMIT = 5
    hparams.mixture_num = 15
    hparams.beam_size = 1
    hparams.connection_dropout = 0.2
    hparams.attention_dropout = 0.
    hparams.diversity_factor = 0.
    hparams.acc_factor = 1.
    return hparams

@Registry.dataset('weibo_keywords_v2')
def weibo_keyword():
    train_data = TabSepFileDataset(os.path.join(config.DATA_PATH, 'weibo_keyword_v2', 'train_data'))
    valid_data = TabSepFileDataset(os.path.join(config.DATA_PATH, 'weibo_keyword_v2', 'valid_data'))
    test_data = TabSepFileDataset(os.path.join(config.DATA_PATH, 'weibo_keyword_v2', 'test_data'))
    return train_data, valid_data, test_data

@Registry.translator('weibo_trans')
class WeiboKeywordCharTranslator(Translator):
    def __init__(self, max_len: int = 30):
        super(WeiboKeywordCharTranslator, self).__init__()
        self.max_len = max_len
        self.shared_namespace = "tokens"

    @staticmethod
    def filter_split_str(sent: str) -> List[str]:
        return list(filter(lambda x: x not in ("", ' ', '\t'), list(sent)))

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        if len(example) != 4:
            logging.warning('Skip invalid line: %s' % str(example))
            return

        src, _, _, tgt  = example
        src = self.filter_split_str(src)[:self.max_len]
        tgt = [START_SYMBOL] + self.filter_split_str(tgt)[:self.max_len] + [END_SYMBOL]

        tokens = src + tgt
        for tok in tokens:
            yield self.shared_namespace, tok

    def to_tensor(self, example):
        assert self.vocab is not None

        src, kwds, conds, tgt = example
        src = self.filter_split_str(src)[:self.max_len]
        tgt = [START_SYMBOL] + self.filter_split_str(tgt)[:self.max_len] + [END_SYMBOL]
        kwds = self.filter_split_str("".join(kwds.split(','))) + [END_SYMBOL]
        conds = self.filter_split_str("".join(conds.split(','))) + [END_SYMBOL]

        def tokens_to_id_vector(token_list: List[str]) -> torch.Tensor:
            ids = [self.vocab.get_token_index(tok, self.shared_namespace) for tok in token_list]
            return torch.tensor(ids, dtype=torch.long)

        tensors = map(tokens_to_id_vector, (src, tgt, kwds, conds))
        return dict(zip(("source_tokens", "target_tokens", "src_keyword_tokens", "tgt_keyword_tokens"), tensors))

    def batch_tensor(self, tensors: List[Mapping[str, torch.Tensor]]):
        assert len(tensors) > 0
        keys = tensors[0].keys()

        tensor_list_by_keys = dict((k, []) for k in keys)
        for instance in tensors:
            for k, tensor in instance.items():
                assert tensor.ndimension() == 1
                tensor_list_by_keys[k].append(tensor)

        batched_tensor = dict(
            (
                k,
                pad_sequence(tlist, batch_first=True,
                             padding_value=self.vocab.get_token_index(PADDING_TOKEN, self.shared_namespace))
            )
            for k, tlist in tensor_list_by_keys.items()
        )

        return batched_tensor

def get_model(hparams, vocab: NSVocabulary):
    kwd_enc = TransformerEncoder(input_dim=hparams.emb_sz,
                                 num_layers=hparams.num_layers,
                                 num_heads=hparams.num_heads,
                                 feedforward_hidden_dim=hparams.emb_sz,
                                 feedforward_dropout=hparams.connection_dropout,
                                 attention_dropout=hparams.attention_dropout,
                                 residual_dropout=hparams.connection_dropout,
                                 )

    src_enc = TransformerEncoder(input_dim=hparams.emb_sz,
                                 num_layers=hparams.num_layers,
                                 num_heads=hparams.num_heads,
                                 feedforward_hidden_dim=hparams.emb_sz,
                                 feedforward_dropout=hparams.connection_dropout,
                                 attention_dropout=hparams.attention_dropout,
                                 residual_dropout=hparams.connection_dropout,
                                 )

    decoder = Decoder([
        DecoderLayer(input_dim=hparams.emb_sz,
                     num_heads=hparams.num_heads,
                     feedforward_hidden_dim=hparams.emb_sz,
                     feedforward_dropout=hparams.connection_dropout,
                     residual_dropout=hparams.connection_dropout,
                     attention_dropout=hparams.attention_dropout,
                     ),
        DecoderLayer(input_dim=hparams.emb_sz,
                     num_heads=hparams.num_heads,
                     feedforward_hidden_dim=hparams.emb_sz,
                     feedforward_dropout=hparams.connection_dropout,
                     residual_dropout=hparams.connection_dropout,
                     attention_dropout=hparams.attention_dropout,
                     ),
    ])
    embedding = Embedding(num_embeddings=vocab.get_vocab_size(), embedding_dim=hparams.emb_sz)
    projection_layer = MoSProjection(hparams.mixture_num, decoder.hidden_dim, vocab.get_vocab_size())

    model = KeywordConditionedTransformer(vocab=vocab,
                                          source_encoder=src_enc,
                                          source_keyword_encoder=kwd_enc,
                                          decoder=decoder,
                                          source_embedding=embedding,
                                          target_embedding=embedding,
                                          target_namespace="tokens",
                                          start_symbol=START_SYMBOL,
                                          eos_symbol=END_SYMBOL,
                                          max_decoding_step=hparams.max_decoding_len,
                                          output_projection_layer=projection_layer,
                                          output_is_logit=False,
                                          beam_size=hparams.beam_size,
                                          )

    return model

def main():
    from training.trial_bot.trial_bot import TrialBot
    import sys
    args = sys.argv[1:] + ['--dataset', 'weibo_keywords_v2', '--translator', 'weibo_trans']
    parser = TrialBot.get_default_parser()
    args = parser.parse_args(args)

    bot = TrialBot(trial_name="keyword_gen_cond", get_model_func=get_model, args=args)
    bot.run()

if __name__ == '__main__':
    main()
