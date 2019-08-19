from typing import List, Generator, Tuple, Mapping, Optional
import os.path
import config
import torch.nn
from collections import defaultdict
from allennlp.modules import Embedding
from models.transformer.encoder import TransformerEncoder
from models.modules.mixture_softmax import MoSProjection
from models.keyword_conditioned_gen.insertion_training_orders import KwdUniformOrder
from models.keyword_conditioned_gen.keyword_insertion_transformer import KeywordInsertionTransformer
from models.transformer.insertion_decoder import InsertionDecoder

from trialbot.data import Translator, NSVocabulary, START_SYMBOL, END_SYMBOL, PADDING_TOKEN, TabSepFileDataset
from trialbot.training import Registry, TrialBot, Events
from trialbot.training.updater import TrainingUpdater, TestingUpdater
from trialbot.data import Iterator, RandomIterator
from trialbot.utils.move_to_device import move_to_device

import logging

@Registry.hparamset()
def weibo_keyword_ins():
    hparams = config.common_settings()
    hparams.emb_sz = 300
    hparams.batch_sz = 100
    hparams.num_enc_layers = 2
    hparams.num_dec_layers = 1
    hparams.num_heads = 6
    hparams.max_decoding_len = 30
    hparams.ADAM_LR = 1e-5
    hparams.TRAINING_LIMIT = 100
    hparams.beam_size = 1
    hparams.connection_dropout = 0.2
    hparams.attention_dropout = 0.
    hparams.diversity_factor = 0.
    hparams.acc_factor = 1.
    hparams.MIN_VOCAB_FREQ = {"tokens": 20}
    hparams.slot_loss = True

    hparams.joint_word_location = False
    hparams.vocab_logit_bias = True
    hparams.mixture_num = 10
    hparams.span_end_penalty = .4
    hparams.num_slot_transformer_layers = 0
    return hparams

@Registry.hparamset()
def weibo_ins_large():
    hparams = weibo_keyword_ins()
    hparams.num_enc_layers = 4
    hparams.num_dec_layers = 4
    return hparams

@Registry.hparamset()
def weibo_ins_large_slot_trans():
    hparams = weibo_ins_large()
    hparams.batch_sz = 64
    hparams.num_slot_transformer_layers = 1
    return hparams

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

@Registry.translator('kwd_ins_char')
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

        yield self.shared_namespace, KwdUniformOrder.END_OF_SPAN_TOKEN

        tokens = src + tgt
        for tok in tokens:
            yield self.shared_namespace, tok

    def to_tensor(self, example):
        assert self.vocab is not None

        src, skwds, tgt, tkwds = example
        src = self.filter_split_str(src)[:self.max_len]
        tgt = self.filter_split_str(tgt)[:self.max_len]
        skwds = self.filter_split_str(skwds)[:self.max_len]
        tkwds = self.filter_split_str(tkwds)[:self.max_len]

        def tokens_to_id_vector(token_list: List[str]) -> torch.Tensor:
            ids = [self.vocab.get_token_index(tok, self.shared_namespace) for tok in token_list]
            return torch.tensor(ids, dtype=torch.long)

        tensors = map(tokens_to_id_vector, (src, tgt, skwds, tkwds))
        return dict(zip(("source_tokens", "target_tokens", "src_keyword_tokens", "tgt_keyword_tokens"), tensors))

    def batch_tensor(self, tensors: List[Mapping[str, torch.Tensor]]):
        assert len(tensors) > 0
        tensor_list_by_keys = defaultdict(list)
        for instance in tensors:
            for k, tensor in instance.items():
                assert tensor.ndimension() == 1
                tensor_list_by_keys[k].append(tensor)

        # discard batch because every source
        return tensor_list_by_keys

def get_model(hparams, vocab: NSVocabulary):
    embedding = Embedding(vocab.get_vocab_size(), hparams.emb_sz)

    encoder = TransformerEncoder(input_dim=hparams.emb_sz,
                                 num_layers=hparams.num_enc_layers,
                                 num_heads=hparams.num_heads,
                                 feedforward_hidden_dim=hparams.emb_sz,
                                 feedforward_dropout=hparams.connection_dropout,
                                 residual_dropout=hparams.connection_dropout,
                                 attention_dropout=hparams.attention_dropout,
                                 )

    decoder = InsertionDecoder(input_dim=hparams.emb_sz,
                               num_layers=hparams.num_dec_layers,
                               num_heads=hparams.num_heads,
                               feedforward_hidden_dim=hparams.emb_sz,
                               feedforward_dropout=hparams.connection_dropout,
                               residual_dropout=hparams.connection_dropout,
                               attention_dropout=hparams.attention_dropout,
                               )

    if hparams.joint_word_location:
        joint_proj = MoSProjection(mixture_num=hparams.mixture_num,
                                   input_dim=decoder.output_dim,
                                   output_dim=vocab.get_vocab_size(),
                                   flatten_softmax=True)
    else:
        slot_proj = torch.nn.Sequential(
            torch.nn.Linear(decoder.output_dim, 1),
            torch.nn.Softmax(dim=-2),
        )
        word_proj = MoSProjection(mixture_num=hparams.mixture_num,
                                  input_dim=decoder.output_dim,
                                  output_dim=vocab.get_vocab_size(),
                                  flatten_softmax=False)
        joint_proj = (slot_proj, word_proj)

    slot_trans = TransformerEncoder(input_dim=decoder.output_dim,
                                    num_layers=hparams.num_slot_transformer_layers,
                                    feedforward_hidden_dim=hparams.emb_sz,
                                    feedforward_dropout=hparams.connection_dropout,
                                    residual_dropout=hparams.connection_dropout,
                                    attention_dropout=hparams.attention_dropout,
                                    ) if hparams.num_slot_transformer_layers > 0 else None

    bias_mapper = None
    if hparams.vocab_logit_bias:
        bias_mapper = torch.nn.Linear(decoder.output_dim, vocab.get_vocab_size())

    model = KeywordInsertionTransformer(vocab=vocab,
                                        encoder=encoder,
                                        decoder=decoder,
                                        src_embedding=embedding,
                                        tgt_embedding=embedding,
                                        joint_projection=joint_proj,
                                        vocab_bias_mapper=bias_mapper,
                                        use_bleu=True,
                                        target_namespace="tokens",
                                        start_symbol=START_SYMBOL,
                                        eos_symbol=END_SYMBOL,
                                        span_ends_symbol=KwdUniformOrder.END_OF_SPAN_TOKEN,
                                        max_decoding_step=hparams.max_decoding_len,
                                        span_end_penalty=hparams.span_end_penalty,
                                        slot_transform=slot_trans,
                                        )

    return model

class InsTrainingUpdater(TrainingUpdater):
    def __init__(self, *args, **kwargs):
        super(InsTrainingUpdater, self).__init__(*args, **kwargs)
        self._order: Optional['KwdUniformOrder'] = None

    @classmethod
    def from_bot(cls, bot: TrialBot):
        self = bot
        args, hparams, model = self.args, self.hparams, self.model
        logger = self.logger

        if hasattr(hparams, "OPTIM") and hparams.OPTIM == "SGD":
            logger.info(f"Using SGD optimzer with lr={hparams.SGD_LR}")
            optim = torch.optim.SGD(model.parameters(), hparams.SGD_LR)
        else:
            logger.info(f"Using Adam optimzer with lr={hparams.ADAM_LR} and beta={str(hparams.ADAM_BETAS)}")
            optim = torch.optim.Adam(model.parameters(), hparams.ADAM_LR, hparams.ADAM_BETAS)

        device = args.device
        dry_run = args.dry_run
        repeat_iter = not args.debug
        shuffle_iter = not args.debug
        iterator = RandomIterator(self.train_set, self.hparams.batch_sz, self.translator,
                                  shuffle=shuffle_iter, repeat=repeat_iter)
        if args.debug and args.skip:
            iterator.reset(args.skip)

        updater = cls(model, iterator, optim, device, dry_run)

        eosid = bot.vocab.get_token_index(KwdUniformOrder.END_OF_SPAN_TOKEN)
        updater._order = KwdUniformOrder(eosid, hparams.slot_loss)
        return updater

    def update_epoch(self):
        model, optim, iterator = self._models[0], self._optims[0], self._iterators[0]
        if iterator.is_new_epoch:
            self.stop_epoch()

        device = self._device
        model.train()
        optim.zero_grad()

        batch = next(iterator)
        keys = ("source_tokens", "target_tokens", "src_keyword_tokens", "tgt_keyword_tokens")
        srcs, tgts, skwds, tkwds = list(map(batch.get, keys))
        dec_inp, locs, contents, weights = next(self._order(tkwds, tgts))

        if device >= 0:
            srcs = move_to_device(srcs, device)
            dec_inp = move_to_device(dec_inp, device)
            locs = move_to_device(locs, device)
            contents = move_to_device(contents, device)
            weights = move_to_device(weights, device)

        targets = (locs, contents, weights)
        output = model(srcs, dec_inp, targets)

        if not self._dry_run:
            loss = output["loss"]
            loss.backward()
            optim.step()

        return output

class InsTestingUpdater(TestingUpdater):
    def update_epoch(self):
        model, iterator, device = self._models[0], self._iterators[0], self._device
        model.eval()
        batch = next(iterator)
        keys = ("source_tokens", "target_tokens", "src_keyword_tokens", "tgt_keyword_tokens")
        srcs, tgts, skwds, tkwds = list(map(batch.get, keys))
        if iterator.is_new_epoch:
            self.stop_epoch()

        if device >= 0:
            srcs = move_to_device(srcs, device)
            tkwds = move_to_device(tkwds, device)
            tgts = move_to_device(tgts, device)

        output = model(srcs, tkwds, references=tgts)
        return output

    @classmethod
    def from_bot(cls, bot: TrialBot) -> 'TestingUpdater':
        self = bot
        args, model = self.args, self.model
        device = args.device

        hparams, model = self.hparams, self.model
        iterator = RandomIterator(self.test_set, hparams.batch_sz, self.translator, shuffle=False, repeat=False)
        updater = cls(model, iterator, None, device)
        return updater

def main():
    import sys
    args = sys.argv[1:]
    if '--dataset' not in sys.argv:
        args += ['--dataset', 'weibo_keywords_v3']
    if '--translator' not in sys.argv:
        args += ['--translator', 'kwd_ins_char']

    parser = TrialBot.get_default_parser()
    args = parser.parse_args(args)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.INFO)

    bot = TrialBot(trial_name="ins_trans", get_model_func=get_model, args=args)
    from trialbot.training.extensions import every_epoch_model_saver
    bot._engine.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)
    if args.test:
        bot.updater = InsTestingUpdater.from_bot(bot)
    else:
        bot.updater = InsTrainingUpdater.from_bot(bot)
    bot.run()

if __name__ == '__main__':
    main()
