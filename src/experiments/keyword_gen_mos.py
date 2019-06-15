import allennlp.data
import allennlp.modules
import config
import data_adapter

from models.transformer.keyword_constrained_seq2seq import KeywordConstrainedTransformer
from models.transformer.encoder import TransformerEncoder
from models.transformer.decoder import TransformerDecoder
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from models.modules.mixture_softmax import MoSProjection

import training.exp_runner as exp_runner

@config.register_hparams
def weibo_keyword_mos():
    hparams = config.common_settings()
    hparams.emb_sz = 300
    hparams.batch_sz = 50
    hparams.num_layers = 2
    hparams.num_heads = 8
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

@config.register_hparams
def weibo_keyword_beam_search():
    hparams = weibo_keyword_mos()
    hparams.beam_size = 6
    hparams.diversity_factor = .9
    hparams.acc_factor = 0.
    return hparams

@config.register_hparams
def weibo_keyword_a01():
    hparams = weibo_keyword_mos()
    hparams.alpha = .1
    return hparams

@config.register_hparams
def weibo_keyword_a001():
    hparams = weibo_keyword_mos()
    hparams.alpha = .01
    return hparams

@config.register_hparams
def weibo_keyword_a10():
    hparams = weibo_keyword_mos()
    hparams.alpha = 10.
    return hparams

@config.register_hparams
def weibo_keyword_a100():
    hparams = weibo_keyword_mos()
    hparams.alpha = 100.
    return hparams

@config.register_hparams
def weibo_keyword_m05():
    hparams = weibo_keyword_mos()
    hparams.margin = .5
    return hparams

@config.register_hparams
def weibo_keyword_m01():
    hparams = weibo_keyword_mos()
    hparams.margin = .1
    return hparams

@config.register_hparams
def weibo_keyword_m001():
    hparams = weibo_keyword_mos()
    hparams.margin = .01
    return hparams

@config.register_hparams
def weibo_keyword_m2():
    hparams = weibo_keyword_mos()
    hparams.margin = 2
    return hparams

@config.register_hparams
def weibo_keyword_m10():
    hparams = weibo_keyword_mos()
    hparams.margin = 10
    return hparams

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

if __name__ == '__main__':
    try:
        exp_runner.run('keyword_mos3', get_model)

    except KeyboardInterrupt:
        pass
