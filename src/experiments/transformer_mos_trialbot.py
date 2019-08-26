from typing import List, Generator, Tuple, Mapping, Optional
import os.path
import config
import torch.nn
from collections import defaultdict
from allennlp.modules import Embedding

from trialbot.data import Translator, NSVocabulary, START_SYMBOL, END_SYMBOL, PADDING_TOKEN, TabSepFileDataset
from trialbot.training import Registry, TrialBot, Events
from trialbot.training.updater import TrainingUpdater, TestingUpdater
from trialbot.data import Iterator, RandomIterator

import logging

from models.transformer.parallel_seq2seq import ParallelSeq2Seq
from models.transformer.encoder import TransformerEncoder
from models.transformer.decoder import TransformerDecoder
from models.modules.mixture_softmax import MoSProjection

import training.exp_runner as exp_runner

@Registry.dataset('small_keywords_v3')
def weibo_keyword():
    train_data = TabSepFileDataset(os.path.join(config.DATA_PATH, 'weibo', 'small_keywords_v3', 'train_data'))
    valid_data = TabSepFileDataset(os.path.join(config.DATA_PATH, 'weibo', 'small_keywords_v3', 'valid_data'))
    test_data = TabSepFileDataset(os.path.join(config.DATA_PATH, 'weibo', 'small_keywords_v3', 'test_data'))
    return train_data, valid_data, test_data

@Registry.dataset('weibo_keywords_v3')
def weibo_keyword():
    train_data = TabSepFileDataset(os.path.join(config.DATA_PATH, 'weibo_keywords_v3', 'train_data'))
    valid_data = TabSepFileDataset(os.path.join(config.DATA_PATH, 'weibo_keywords_v3', 'valid_data'))
    test_data = TabSepFileDataset(os.path.join(config.DATA_PATH, 'weibo_keywords_v3', 'test_data'))
    return train_data, valid_data, test_data

@Registry.translator('char_trans')
class KeywordCharInsTranslator(Translator):
    def __init__(self, max_len: int = 30):
        super(KeywordCharInsTranslator, self).__init__()
        self.max_len = max_len
        self.shared_namespace = "tokens"

    @staticmethod
    def filter_split_str(sent: str) -> List[str]:
        return list(filter(lambda x: x not in ("", ' ', '\t'), list(sent)))

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        if len(example) != 4:
            logging.warning('Skip invalid line: %s' % str(example))
            return

        src, _, tgt, _  = example
        src = self.filter_split_str(src)[:self.max_len]
        tgt = [START_SYMBOL] + self.filter_split_str(tgt)[:self.max_len] + [END_SYMBOL]

        tokens = src + tgt
        for tok in tokens:
            yield self.shared_namespace, tok

    def to_tensor(self, example):
        assert self.vocab is not None

        src, skwds, tgt, tkwds = example
        src = self.filter_split_str(src)[:self.max_len]
        tgt = self.filter_split_str(tgt)[:self.max_len]

        def tokens_to_id_vector(token_list: List[str]) -> torch.Tensor:
            ids = [self.vocab.get_token_index(tok, self.shared_namespace) for tok in token_list]
            return torch.tensor(ids, dtype=torch.long)

        tensors = map(tokens_to_id_vector, (src, tgt))
        return dict(zip(("source_tokens", "target_tokens"), tensors))

    def batch_tensor(self, tensors: List[Mapping[str, torch.Tensor]]):
        assert len(tensors) > 0
        tensor_list_by_keys = defaultdict(list)
        for instance in tensors:
            for k, tensor in instance.items():
                assert tensor.ndimension() == 1
                tensor_list_by_keys[k].append(tensor)

        # discard batch because every source
        return tensor_list_by_keys


@config.register_hparams
def weibo_trans_mos():
    hparams = config.common_settings()
    hparams.emb_sz = 300
    hparams.batch_sz = 128
    hparams.num_layers = 4
    hparams.num_heads = 8
    hparams.feedforward_dropout = .2
    hparams.max_decoding_len = 30
    hparams.ADAM_LR = 1e-4
    hparams.TRAINING_LIMIT = 20
    hparams.mixture_num = 15
    hparams.beam_size = 1
    hparams.diversity_factor = 0.
    hparams.acc_factor = 1.
    return hparams

@config.register_hparams
def weibo_trans_mos3():
    hparams = weibo_trans_mos()
    hparams.num_layers = 2
    return hparams

@config.register_hparams
def weibo_trans_mos3_bs6():
    hparams = weibo_trans_mos3()
    hparams.beam_size = 6
    hparams.diversity_factor = 0.
    hparams.acc_factor = 1.
    return hparams

@config.register_hparams
def weibo_trans_mos3_bs6_div1():
    hparams = weibo_trans_mos3_bs6()
    hparams.diversity_factor = 1.
    return hparams

@config.register_hparams
def weibo_trans_mos3_bs6_acc0():
    hparams = weibo_trans_mos3_bs6()
    hparams.acc_factor = 0.
    return hparams

@config.register_hparams
def weibo_trans_mos3_bs6_acc0_div09():
    hparams = weibo_trans_mos3_bs6()
    hparams.acc_factor = 0.
    hparams.diversity_factor = .9
    return hparams

def get_model(hparams, vocab: NSVocabulary):
    encoder = TransformerEncoder(input_dim=hparams.emb_sz,
                                 num_layers=hparams.num_layers,
                                 num_heads=hparams.num_heads,
                                 feedforward_hidden_dim=hparams.emb_sz,)
    decoder = TransformerDecoder(input_dim=hparams.emb_sz,
                                 num_layers=hparams.num_layers,
                                 num_heads=hparams.num_heads,
                                 feedforward_hidden_dim=hparams.emb_sz,
                                 feedforward_dropout=hparams.feedforward_dropout,)
    embedding = Embedding(num_embeddings=vocab.get_vocab_size(),
                          embedding_dim=hparams.emb_sz)
    projection_layer = MoSProjection(hparams.mixture_num, decoder.hidden_dim, vocab.get_vocab_size())
    model = ParallelSeq2Seq(vocab=vocab,
                            encoder=encoder,
                            decoder=decoder,
                            source_embedding=embedding,
                            target_embedding=embedding,
                            target_namespace='tokens',
                            start_symbol=START_SYMBOL,
                            eos_symbol=END_SYMBOL,
                            max_decoding_step=hparams.max_decoding_len,
                            output_projection_layer=projection_layer,
                            output_is_logit=False,
                            beam_size=hparams.beam_size,
                            diversity_factor=hparams.diversity_factor,
                            accumulation_factor=hparams.acc_factor,
                            )
    return model

if __name__ == '__main__':
    try:
        exp_runner.run('transfomer_mos3', get_model)

    except KeyboardInterrupt:
        pass
