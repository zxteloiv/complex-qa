from typing import Tuple
import allennlp.modules
import config
import os.path
from data_adapter.ns_vocabulary import NSVocabulary, START_SYMBOL, END_SYMBOL
import training.trial_bot.trial_bot as trial_bot
from data_adapter.translators.weibo_keyword_translator import CharBasedWeiboKeywordsTranslator

from models.transformer.keyword_constrained_seq2seq import KeywordConstrainedTransformer
from models.transformer.encoder import TransformerEncoder
from models.transformer.decoder import TransformerDecoder
from models.modules.mixture_softmax import MoSProjection

from data_adapter.dataset import Dataset
from data_adapter.general_datasets.tabular_dataset import TabSepFileDataset

from training.trial_bot.trial_registry import Registry

@Registry.hparamset()
def weibo_keyword_mos():
    hparams = config.common_settings()
    hparams.emb_sz = 300
    hparams.batch_sz = 50
    hparams.num_layers = 2
    hparams.num_heads = 8
    hparams.feedforward_dropout = .2
    hparams.max_decoding_len = 30
    hparams.ADAM_LR = 1e-3
    hparams.TRAINING_LIMIT = 5
    hparams.mixture_num = 15
    hparams.beam_size = 1
    hparams.diversity_factor = .9
    hparams.acc_factor = 0.

    hparams.margin = 1.
    hparams.alpha = 1.
    return hparams

@Registry.dataset()
def weibo_keywords() -> Tuple[Dataset, Dataset, Dataset]:
    """
DATASETS["weibo_keywords"] = DatasetPath(
    train_path=os.path.join(DATA_PATH, 'weibo_keywords', 'train_data'),
    dev_path=os.path.join(DATA_PATH, 'weibo_keywords', 'valid_data'),
    test_path=os.path.join(DATA_PATH, 'weibo_keywords', 'test_data'),
)
    """
    train_set = TabSepFileDataset(os.path.join(config.DATA_PATH, 'weibo_keywords', 'train_data'))
    dev_set = TabSepFileDataset(os.path.join(config.DATA_PATH, 'weibo_keywords', 'dev_data'))
    test_set = TabSepFileDataset(os.path.join(config.DATA_PATH, 'weibo_keywords', 'test_data'))
    return train_set, dev_set, test_set


@Registry.translator("weibo_keyword_translator")
class CharBasedWeiboKeywordsTranslatorForAllenNLP(CharBasedWeiboKeywordsTranslator):
    def batch_tensor(self, tensors):
        batch_tensors = super(CharBasedWeiboKeywordsTranslatorForAllenNLP, self).batch_tensor(tensors)
        allennlp_model_tensors = dict()
        for k, v in batch_tensors.items():
            allennlp_model_tensors[k] = {"tokens": v}
        return allennlp_model_tensors

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
        bot = trial_bot.TrialBot(get_model_func=get_model)
        bot.run()

    except KeyboardInterrupt:
        pass
