import allennlp.data
import allennlp.modules
import config
import data_adapter
import torch.nn as nn

from models.transformer.parallel_seq2seq import ParallelSeq2Seq
from models.transformer.encoder import TransformerEncoder
from models.transformer.decoder import TransformerDecoder
from allennlp.common.util import START_SYMBOL, END_SYMBOL

import training.exp_runner as exp_runner

@config.register_hparams
def weibo_trans_tied():
    hparams = config.common_settings()
    hparams.emb_sz = 300
    hparams.batch_sz = 128
    hparams.num_layers = 2
    hparams.num_heads = 8
    hparams.feedforward_dropout = .2
    hparams.max_decoding_len = 30
    hparams.ADAM_LR = 1e-4
    hparams.TRAINING_LIMIT = 20
    hparams.beam_size = 6
    hparams.diversity_factor = 1.0
    hparams.acc_factor = 1.0
    return hparams

@config.register_hparams
def weibo_trans_tied_large():
    hparams = weibo_trans_tied()
    hparams.emb_sz = 1024
    hparams.batch_sz = 50
    return hparams

@config.register_hparams
def weibo_trans_tied_large_greedy():
    hparams = weibo_trans_tied_large()
    hparams.beam_size = 1
    return hparams

@config.register_hparams
def weibo_trans_tied_large_no_diverse():
    hparams = weibo_trans_tied_large()
    hparams.diversity_factor = 0.
    return hparams

@config.register_hparams
def weibo_trans_tied_large_no_acc():
    hparams = weibo_trans_tied_large()
    hparams.acc_factor = 0.
    hparams.diversity_factor = 0.
    return hparams

@config.register_hparams
def weibo_trans_tied_large_diverse_no_acc():
    hparams = weibo_trans_tied_large()
    hparams.acc_factor = 0.
    hparams.diversity_factor = .9
    return hparams

@config.register_hparams
def weibo_trans_tied_greedy():
    hparams = weibo_trans_tied()
    hparams.beam_size = 1
    return hparams

@config.register_hparams
def weibo_trans_tied_no_diverse():
    hparams = weibo_trans_tied()
    hparams.diversity_factor = 0.
    return hparams

@config.register_hparams
def weibo_trans_tied_no_acc():
    hparams = weibo_trans_tied_no_diverse()
    hparams.acc_factor = 0.
    return hparams

@config.register_hparams
def weibo_trans_tied_diverse_no_acc():
    hparams = weibo_trans_tied()
    hparams.acc_factor = 0.
    hparams.diversity_factor = .9
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
    projection_layer = nn.Linear(decoder.hidden_dim, vocab.get_vocab_size())
    projection_layer.weight = embedding.weight
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
                            output_is_logit=True,
                            beam_size=hparams.beam_size,
                            diversity_factor=hparams.diversity_factor,
                            accumulation_factor=hparams.acc_factor,
                            )
    return model

if __name__ == '__main__':
    try:
        exp_runner.run('transfomer_tied_dec', get_model)

    except KeyboardInterrupt:
        pass
