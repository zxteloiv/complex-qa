from typing import List, Generator, Tuple, Mapping
import os.path
import config
import torch
from torch.nn.utils.rnn import pad_sequence
from allennlp.modules import Embedding
from allennlp.modules.attention import BilinearAttention
from utils.nn import AllenNLPAttentionWrapper
from training.trial_bot.data.ns_vocabulary import NSVocabulary
from training.trial_bot.data.general_datasets.tabular_dataset import TabSepFileDataset
from models.keyword_conditioned_gen.keyword_seq2seq import Seq2KeywordSeq, StackedEncoder
from models.modules.stacked_rnn_cell import StackedLSTMCell
from models.transformer.encoder import TransformerEncoder
from models.modules.mixture_softmax import MoSProjection
from training.trial_bot.data.ns_vocabulary import START_SYMBOL, END_SYMBOL, PADDING_TOKEN
from training.trial_bot.trial_registry import Registry
from training.trial_bot.data.translator import Translator
import logging

@Registry.hparamset()
def weibo_keyword_gen():
    hparams = config.common_settings()
    hparams.emb_sz = 300
    hparams.batch_sz = 50
    hparams.num_enc_layers = 2
    hparams.num_dec_layers = 1
    hparams.num_heads = 6
    hparams.max_decoding_len = 30
    hparams.ADAM_LR = 1e-5
    hparams.TRAINING_LIMIT = 20
    hparams.mixture_num = 15
    hparams.beam_size = 1
    hparams.connection_dropout = 0.2
    hparams.attention_dropout = 0.
    hparams.diversity_factor = 0.
    hparams.acc_factor = 1.
    return hparams

@Registry.dataset('small_keywords_v2')
def weibo_keyword():
    train_data = TabSepFileDataset(os.path.join(config.DATA_PATH, 'weibo', 'small_keywords_v2', 'train_data'))
    valid_data = TabSepFileDataset(os.path.join(config.DATA_PATH, 'weibo', 'small_keywords_v2', 'valid_data'))
    test_data = TabSepFileDataset(os.path.join(config.DATA_PATH, 'weibo', 'small_keywords_v2', 'test_data'))
    return train_data, valid_data, test_data

@Registry.dataset('weibo_keywords_v2')
def weibo_keyword():
    train_data = TabSepFileDataset(os.path.join(config.DATA_PATH, 'weibo_keywords_v2', 'train_data'))
    valid_data = TabSepFileDataset(os.path.join(config.DATA_PATH, 'weibo_keywords_v2', 'valid_data'))
    test_data = TabSepFileDataset(os.path.join(config.DATA_PATH, 'weibo_keywords_v2', 'test_data'))
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
        kwds = self.filter_split_str("".join(kwds.split(' ')))[:self.max_len] + [END_SYMBOL]
        conds = self.filter_split_str("".join(conds.split(' ')))[:self.max_len] + [END_SYMBOL]

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
    src_enc = StackedEncoder(
        [
            TransformerEncoder(input_dim=hparams.emb_sz,
                               num_layers=hparams.num_enc_layers,
                               num_heads=hparams.num_heads,
                               feedforward_hidden_dim=hparams.emb_sz,
                               feedforward_dropout=hparams.connection_dropout,
                               attention_dropout=hparams.attention_dropout,
                               residual_dropout=hparams.connection_dropout,
                               )
        ], input_size=hparams.emb_sz, output_size=hparams.emb_sz,
    )

    enc_attn = AllenNLPAttentionWrapper(BilinearAttention(vector_dim=hparams.emb_sz, matrix_dim=hparams.emb_sz),
                                        attn_dropout=hparams.attention_dropout)
    dec_hist_attn = AllenNLPAttentionWrapper(BilinearAttention(vector_dim=hparams.emb_sz, matrix_dim=hparams.emb_sz),
                                             attn_dropout=hparams.attention_dropout)

    decoder = StackedLSTMCell(input_dim=(hparams.emb_sz * 4),   # input + keyword + enc_attn + dec_hist_attn
                              hidden_dim=hparams.emb_sz,
                              n_layers=hparams.num_dec_layers,
                              intermediate_dropout=hparams.connection_dropout,)

    embedding = Embedding(num_embeddings=vocab.get_vocab_size(), embedding_dim=hparams.emb_sz)
    projection_layer = MoSProjection(hparams.mixture_num,
                                     hparams.emb_sz * 3,    # input + enc_attn + dec_hist_attn
                                     vocab.get_vocab_size())

    model = Seq2KeywordSeq(vocab=vocab,
                           encoder=src_enc,
                           decoder=decoder,
                           word_projection=projection_layer,
                           source_embedding=embedding,
                           target_embedding=embedding,
                           target_namespace="tokens",
                           start_symbol=START_SYMBOL,
                           eos_symbol=END_SYMBOL,
                           max_decoding_step=hparams.max_decoding_len,
                           enc_attention=enc_attn,
                           dec_hist_attn=dec_hist_attn,
                           scheduled_sampling_ratio=.1,
                           intermediate_dropout=hparams.connection_dropout,
                           output_is_logit=False,
                           )

    return model

def main():
    from training.trial_bot.trial_bot import TrialBot, Events
    import sys
    import json
    args = sys.argv[1:] + ['--translator', 'weibo_trans']
    if '--dataset' not in sys.argv:
        args += ['--dataset', 'weibo_keywords_v2']

    parser = TrialBot.get_default_parser()
    args = parser.parse_args(args)

    bot = TrialBot(trial_name="keyword_gen_s2s", get_model_func=get_model, args=args)
    @bot.attach_extension(Events.ITERATION_COMPLETED)
    def ext_metrics(bot: TrialBot):
        if bot.state.iteration % 40 == 0:
            metrics = bot.model.get_metrics()
            bot.logger.info("metrics: " + json.dumps(metrics))

    @bot.attach_extension(Events.EPOCH_COMPLETED)
    def epoch_clean_metrics(bot: TrialBot):
        metrics = bot.model.get_metrics(reset=True)
        bot.logger.info("Epoch metrics: " + json.dumps(metrics))

    bot.run()

if __name__ == '__main__':
    main()
