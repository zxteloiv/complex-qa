import allennlp.data
import allennlp.modules
import config
import data_adapter

from models.transformer.parallel_seq2seq import ParallelSeq2Seq
from models.transformer.encoder import TransformerEncoder
from models.transformer.decoder import TransformerDecoder
from allennlp.common.util import START_SYMBOL, END_SYMBOL

import training.exp_runner as exp_runner

@config.register_hparams
def weibo_trans_for_alibaba():
    hparams = config.common_settings()
    hparams.emb_sz = 300
    hparams.batch_sz = 128
    hparams.num_layers = 6
    hparams.num_heads = 8
    hparams.feedforward_dropout = .3
    hparams.max_decoding_len = 30
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
    model = ParallelSeq2Seq(vocab=vocab,
                            encoder=encoder,
                            decoder=decoder,
                            source_embedding=embedding,
                            target_embedding=embedding,
                            target_namespace='tokens',
                            start_symbol=START_SYMBOL,
                            eos_symbol=END_SYMBOL,
                            max_decoding_step=hparams.max_decoding_len,
                            )
    return model

if __name__ == '__main__':
    try:
        exp_runner.run('transfomer', get_model)

    except KeyboardInterrupt:
        pass
