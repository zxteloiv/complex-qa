from typing import List, Generator, Tuple, Mapping
import os.path
import torch
from torch.nn.utils.rnn import pad_sequence
from allennlp.modules import Embedding
from trialbot.data.ns_vocabulary import NSVocabulary
from trialbot.data.datasets.tabular_dataset import TabSepFileDataset
from models.keyword_conditioned_gen.keyword_conditioned_transformer import KeywordConditionedTransformer, Decoder, DecoderLayer
from models.transformer.encoder import TransformerEncoder
from models.modules.mixture_softmax import MoSProjection
from trialbot.data.ns_vocabulary import START_SYMBOL, END_SYMBOL, PADDING_TOKEN
from trialbot.training import Registry
from trialbot.training.hparamset import HyperParamSet
from trialbot.data.translator import Translator
import logging
from utils.root_finder import find_root
_DATA_PATH = os.path.join(find_root(), 'data')

@Registry.hparamset()
def weibo_keyword_gen():
    hparams = HyperParamSet.common_settings(find_root())
    hparams.emb_sz = 300
    hparams.batch_sz = 50
    hparams.num_layers = 2
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
    hparams.cross_entropy = True # use cross entropy or not
    hparams.margin = 1.
    return hparams

@Registry.hparamset()
def weibo_keyword_gen_bighead_mloss():
    hparams = weibo_keyword_gen()
    hparams.emb_sz = 300
    hparams.batch_sz = 50
    hparams.num_heads = 10
    hparams.cross_entropy = False # do not use cross entropy but margin loss
    hparams.margin = 1.
    return hparams

@Registry.hparamset()
def weibo_keyword_gen_sgd():
    hparams = weibo_keyword_gen()
    hparams.OPTIM = "SGD"
    hparams.SGD_LR = 1e-4
    return hparams

@Registry.hparamset()
def weibo_keyword_gen_bighead_m10():
    hparams = weibo_keyword_gen_bighead_mloss()
    hparams.margin = 10.
    return hparams

@Registry.dataset('weibo_keywords_v3')
def weibo_keyword():
    train_data = TabSepFileDataset(os.path.join(_DATA_PATH, 'weibo_keywords_v3', 'train_data'))
    valid_data = TabSepFileDataset(os.path.join(_DATA_PATH, 'weibo_keywords_v3', 'valid_data'))
    test_data = TabSepFileDataset(os.path.join(_DATA_PATH, 'weibo_keywords_v3', 'test_data'))
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
    projection_layer = MoSProjection(hparams.mixture_num, hparams.emb_sz, vocab.get_vocab_size())

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
                                          use_cross_entropy=hparams.cross_entropy,
                                          margin=hparams.margin,
                                          )

    return model

def main():
    from trialbot.training import TrialBot, Events
    import sys
    import json
    args = sys.argv[1:] + ['--dataset', 'weibo_keywords_v3', '--translator', 'weibo_trans']
    parser = TrialBot.get_default_parser()
    args = parser.parse_args(args)

    bot = TrialBot(trial_name="keyword_gen_cond", get_model_func=get_model, args=args)
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
    main()
