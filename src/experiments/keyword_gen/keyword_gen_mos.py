from typing import List, Generator, Tuple, Mapping
import allennlp.data
import allennlp.modules

from models.keyword_conditioned_gen.keyword_constrained_transformer import KeywordConstrainedTransformer
from models.transformer.encoder import TransformerEncoder
from models.transformer.decoder import TransformerDecoder
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from models.modules.mixture_softmax import MoSProjection

import torch
from torch.nn.utils.rnn import pad_sequence
import os.path
from trialbot.data import Translator, NSVocabulary, START_SYMBOL, END_SYMBOL, PADDING_TOKEN, TabSepFileDataset
from trialbot.training import Registry, TrialBot, Events
from trialbot.training.hparamset import HyperParamSet
from utils.root_finder import find_root
import logging
_DATA_PATH = os.path.join(find_root(), 'data')

@Registry.hparamset()
def weibo_keyword_mos():
    hparams = HyperParamSet.common_settings(find_root())
    hparams.emb_sz = 300
    hparams.batch_sz = 50
    hparams.num_layers = 2
    hparams.num_heads = 6
    hparams.feedforward_dropout = .2
    hparams.max_decoding_len = 30
    hparams.ADAM_LR = 1e-5
    hparams.TRAINING_LIMIT = 5
    hparams.mixture_num = 15
    hparams.beam_size = 1
    hparams.diversity_factor = .9
    hparams.acc_factor = 0.

    hparams.margin = 1.
    hparams.alpha = 1.
    return hparams

@Registry.hparamset()
def weibo_keyword_beam_search():
    hparams = weibo_keyword_mos()
    hparams.beam_size = 6
    hparams.diversity_factor = .9
    hparams.acc_factor = 0.
    return hparams

@Registry.hparamset()
def weibo_keyword_a01n():
    hparams = weibo_keyword_mos()
    hparams.alpha = -.1
    return hparams

@Registry.hparamset()
def weibo_keyword_a1e15n():
    # almost no keywords information is considered
    hparams = weibo_keyword_mos()
    hparams.mixture_num = 10
    hparams.batch_sz = 64
    hparams.TRAINING_LIMIT = 50
    hparams.ADAM_LR = 1e-4
    hparams.alpha = -1e14
    return hparams

@Registry.hparamset()
def weibo_keyword_v3_a1m1():
    # almost no keywords information is considered
    hparams = weibo_keyword_a1e15n()
    hparams.TRAINING_LIMIT = 10
    hparams.ADAM_LR = 1e-5
    hparams.alpha = 1.
    hparams.margin = 1.
    return hparams

@Registry.hparamset()
def weibo_keyword_v3_a1n():
    # almost no keywords information is considered
    hparams = weibo_keyword_v3_a1m1()
    hparams.alpha = -1.
    return hparams

@Registry.hparamset()
def weibo_keyword_a1n():
    hparams = weibo_keyword_mos()
    hparams.alpha = -1.
    return hparams

@Registry.hparamset()
def weibo_keyword_a10n():
    hparams = weibo_keyword_mos()
    hparams.alpha = -10.
    return hparams

@Registry.hparamset()
def weibo_keyword_a100n():
    hparams = weibo_keyword_mos()
    hparams.alpha = -100.
    return hparams

@Registry.hparamset()
def weibo_keyword_a01():
    hparams = weibo_keyword_mos()
    hparams.alpha = .1
    return hparams

@Registry.hparamset()
def weibo_keyword_a001():
    hparams = weibo_keyword_mos()
    hparams.alpha = .01
    return hparams

@Registry.hparamset()
def weibo_keyword_a10():
    hparams = weibo_keyword_mos()
    hparams.alpha = 10.
    return hparams

@Registry.hparamset()
def weibo_keyword_a100():
    hparams = weibo_keyword_mos()
    hparams.alpha = 100.
    return hparams

@Registry.hparamset()
def weibo_keyword_m05():
    hparams = weibo_keyword_mos()
    hparams.margin = .5
    return hparams

@Registry.hparamset()
def weibo_keyword_m01():
    hparams = weibo_keyword_mos()
    hparams.margin = .1
    return hparams

@Registry.hparamset()
def weibo_keyword_m001():
    hparams = weibo_keyword_mos()
    hparams.margin = .01
    return hparams

@Registry.hparamset()
def weibo_keyword_m2():
    hparams = weibo_keyword_mos()
    hparams.margin = 2
    return hparams

@Registry.hparamset()
def weibo_keyword_a10_m01():
    hparams = weibo_keyword_a10()
    hparams.margin = .1
    return hparams

@Registry.hparamset()
def weibo_keyword_a10_m2():
    hparams = weibo_keyword_a10()
    hparams.margin = .1
    return hparams

@Registry.hparamset()
def weibo_keyword_m10():
    hparams = weibo_keyword_mos()
    hparams.margin = 10
    return hparams

@Registry.dataset('small_keywords_v3')
def weibo_keyword():
    train_data = TabSepFileDataset(os.path.join(_DATA_PATH, 'weibo', 'small_keywords_v3', 'train_data'))
    valid_data = TabSepFileDataset(os.path.join(_DATA_PATH, 'weibo', 'small_keywords_v3', 'valid_data'))
    test_data = TabSepFileDataset(os.path.join(_DATA_PATH, 'weibo', 'small_keywords_v3', 'test_data'))
    return train_data, valid_data, test_data

@Registry.dataset('weibo_keywords_v3')
def weibo_keyword():
    train_data = TabSepFileDataset(os.path.join(_DATA_PATH, 'weibo_keywords_v3', 'train_data'))
    valid_data = TabSepFileDataset(os.path.join(_DATA_PATH, 'weibo_keywords_v3', 'valid_data'))
    test_data = TabSepFileDataset(os.path.join(_DATA_PATH, 'weibo_keywords_v3', 'test_data'))
    return train_data, valid_data, test_data

@Registry.translator('weibo_trans_char')
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

        src, _, tgt, _  = example
        src = self.filter_split_str(src)[:self.max_len]
        tgt = [START_SYMBOL] + self.filter_split_str(tgt)[:self.max_len] + [END_SYMBOL]

        tokens = src + tgt
        for tok in tokens:
            yield self.shared_namespace, tok

    def to_tensor(self, example):
        assert self.vocab is not None

        src, kwds, tgt, conds = example
        src = self.filter_split_str(src)[:self.max_len]
        tgt = [START_SYMBOL] + self.filter_split_str(tgt)[:self.max_len] + [END_SYMBOL]
        kwds = self.filter_split_str(kwds)[:self.max_len] + [END_SYMBOL]
        conds = self.filter_split_str(conds)[:self.max_len] + [END_SYMBOL]

        def tokens_to_id_vector(token_list: List[str]) -> torch.Tensor:
            ids = [self.vocab.get_token_index(tok, self.shared_namespace) for tok in token_list]
            return torch.tensor(ids, dtype=torch.long)

        tensors = map(tokens_to_id_vector, (src, tgt, conds))
        return dict(zip(("source_tokens", "target_tokens", "keyword_tokens"), tensors))

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

@Registry.translator('weibo_trans_word')
class WeiboKeywordWordTranslator(WeiboKeywordCharTranslator):
    def __init__(self, max_len: int = 20):
        super(WeiboKeywordWordTranslator, self).__init__(max_len)

    @staticmethod
    def filter_split_str(sent: str) -> List[str]:
        return list(filter(lambda x: x not in ("", ' ', '\t'), sent.split(' ')))

def get_model(hparams, vocab: allennlp.data.Vocabulary):
    encoder = TransformerEncoder(input_dim=hparams.emb_sz,
                                 num_layers=hparams.num_layers,
                                 num_heads=hparams.num_heads,
                                 feedforward_hidden_dim=hparams.emb_sz,)
    decoder = TransformerDecoder(input_dim=hparams.emb_sz,
                                 num_layers=hparams.num_layers,
                                 num_heads=hparams.num_heads,
                                 feedforward_hidden_dim=hparams.emb_sz,
                                 feedforward_dropout=hparams.feedforward_dropout,)
    embedding = allennlp.modules.Embedding(num_embeddings=vocab.get_vocab_size(),
                                           embedding_dim=hparams.emb_sz)
    projection_layer = MoSProjection(hparams.mixture_num, decoder.hidden_dim, vocab.get_vocab_size())
    model = KeywordConstrainedTransformer(vocab=vocab,
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
                                          margin=hparams.margin,
                                          alpha=hparams.alpha,
                                          )
    return model

def main():
    import sys
    import json
    args = sys.argv[1:]
    if '--dataset' not in sys.argv:
        args += ['--dataset', 'weibo_keywords_v3']
    if '--translator' not in sys.argv:
        args += ['--translator', 'weibo_trans_char']

    parser = TrialBot.get_default_parser()
    args = parser.parse_args(args)

    bot = TrialBot(trial_name="keyword_gen_mos", get_model_func=get_model, args=args)
    @bot.attach_extension(Events.ITERATION_COMPLETED)
    def ext_metrics(bot: TrialBot):
        if bot.state.iteration % 40 == 0:
            metrics = bot.model.get_metrics()
            bot.logger.info("metrics: " + json.dumps(metrics))

    @bot.attach_extension(Events.EPOCH_COMPLETED)
    def epoch_clean_metrics(bot: TrialBot):
        metrics = bot.model.get_metrics(reset=True)
        bot.logger.info("Epoch metrics: " + json.dumps(metrics))

    from trialbot.training.extensions import every_epoch_model_saver, legacy_testing_output
    if args.test:
        bot.add_event_handler(Events.ITERATION_COMPLETED, legacy_testing_output, 100)
    else:
        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)

    bot.run()


if __name__ == '__main__':
    try:
        main()

    except KeyboardInterrupt:
        pass
