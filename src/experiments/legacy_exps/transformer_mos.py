import allennlp.data
import allennlp.modules
import config
import data_adapter

from models.transformer.parallel_seq2seq import ParallelSeq2Seq
from models.transformer.encoder import TransformerEncoder
from models.transformer.decoder import TransformerDecoder
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from models.modules.mixture_softmax import MoSProjection

import training.exp_runner as exp_runner

@config.register_hparams
def weibo_trans_mos():
    hparams = config.common_settings()
    hparams.emb_sz = 300
    hparams.batch_sz = 128
    hparams.num_layers = 6
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
