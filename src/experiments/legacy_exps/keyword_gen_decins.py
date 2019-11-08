from typing import List, Generator, Tuple, Mapping, Optional
import os.path
import config
import torch.nn
from collections import defaultdict
from allennlp.modules import Embedding
from models.transformer.encoder import TransformerEncoder
from models.modules.mixture_softmax import MoSProjection
from models.keyword_conditioned_gen.insertion_training_transformation import DecoupledInsTrans
from models.keyword_conditioned_gen.decoupled_insertion_transformer import DecoupledInsertionTransformer
from models.transformer.insertion_decoder import InsertionDecoder

from trialbot.data import Translator, NSVocabulary, TabSepFileDataset
from trialbot.training import Registry, TrialBot, Events
from trialbot.training.updater import TrainingUpdater, TestingUpdater
from trialbot.utils.move_to_device import move_to_device
from sentencepiece import SentencePieceProcessor

import logging

@Registry.hparamset()
def weibo_keyword_ins():
    hparams = config.common_settings()
    hparams.emb_sz = 300
    hparams.batch_sz = 100
    hparams.num_enc_layers = 4
    hparams.num_dec_layers = 4
    hparams.num_heads = 6
    hparams.max_decoding_len = 23
    hparams.ADAM_LR = 1e-5
    hparams.TRAINING_LIMIT = 20
    hparams.beam_size = 1
    hparams.connection_dropout = 0.2
    hparams.attention_dropout = 0.
    hparams.diversity_factor = 0.
    hparams.acc_factor = 1.
    hparams.MIN_VOCAB_FREQ = {"tokens": 20}

    hparams.mixture_num = 10
    hparams.span_end_threshold = 0.5
    hparams.num_slot_transformer_layers = 0
    hparams.num_dual_model_layers = 4

    hparams.alpha1 = 0.5
    hparams.alpha2 = 0.5

    hparams.stammering_window = 2
    hparams.topk_words = 6
    return hparams

@Registry.hparamset()
def weibo_keyword_ins_freegen():
    hparams = weibo_keyword_ins()
    hparams.span_end_threshold = 0.01
    hparams.topk_words = 30
    hparams.stammering_window = 5   # free generation needs stricter rule restrictions
    return hparams

@Registry.hparamset()
def weibo_keyword_ins_prototype():
    hparams = weibo_keyword_ins()
    hparams.span_end_threshold = 0.5
    hparams.topk_words = 30
    hparams.stammering_window = 5   # free generation needs stricter rule restrictions
    hparams.mixture_num = 1     # no MoS used for smaller number of vocab mapping parameters
    hparams.emb_sz = 512    # greater embedding for better expressiveness
    hparams.alpha1 = .1     # 9:1 for word:slot loss
    return hparams

@Registry.hparamset()
def weibo_keyword_ins_prototype_freegen():
    hparams = weibo_keyword_ins_prototype()
    hparams.span_end_threshold = 0.01
    hparams.topk_words = 30
    return hparams

@Registry.hparamset()
def weibo_keyword_ins_quicklearning():
    hparams = weibo_keyword_ins()
    hparams.ADAM_LR = 1e-4
    return hparams

@Registry.hparamset()
def training_debug():
    hparams = weibo_keyword_ins()
    hparams.TRAINING_LIMIT = 1
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

@Registry.translator('kwd_ins_spm')
class KeywordSPMInsTranslator(Translator):
    def __init__(self, max_len: int = 30):
        super().__init__()
        self.max_len = max_len
        self.shared_namespace = "tokens"
        self.spm = SentencePieceProcessor()
        self.spm.Load(os.path.expanduser("~/.complex_qa/sentencepiece_models/weibo5120.model"))

    def filter_split_str(self, sent: str) -> List[str]:
        return self.spm.EncodeAsPieces(sent.replace(" ", ""))

    START_SYMBOL = '<s>'
    END_SYMBOL = '</s>'
    MASK_SYMBOL = '<mask>'

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        if len(example) != 4:
            logging.warning('Skip invalid line: %s' % str(example))
            return

        src, _, tgt, _  = example
        src = self.filter_split_str(src)[:self.max_len]
        tgt = [self.START_SYMBOL] + self.filter_split_str(tgt)[:self.max_len] + [self.END_SYMBOL]

        yield self.shared_namespace, self.MASK_SYMBOL

        tokens = src + tgt
        for tok in tokens:
            yield self.shared_namespace, tok

    def to_tensor(self, example):
        assert self.vocab is not None

        src, skwds, tgt, tkwds = list(map(lambda x: self.filter_split_str(x)[:self.max_len], example))
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

@Registry.translator('kwd_ins_char')
class KeywordCharInsTranslator(KeywordSPMInsTranslator):
    def filter_split_str(self, sent: str):
        return list(sent.replace(" ", ""))

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

    slot_decoder = InsertionDecoder(input_dim=hparams.emb_sz,
                                    num_layers=hparams.num_dec_layers,
                                    num_heads=hparams.num_heads,
                                    feedforward_hidden_dim=hparams.emb_sz,
                                    feedforward_dropout=hparams.connection_dropout,
                                    residual_dropout=hparams.connection_dropout,
                                    attention_dropout=hparams.attention_dropout,
                                    )

    slot_trans = TransformerEncoder(input_dim=slot_decoder.output_dim,
                                    num_layers=hparams.num_slot_transformer_layers,
                                    feedforward_hidden_dim=hparams.emb_sz,
                                    feedforward_dropout=hparams.connection_dropout,
                                    residual_dropout=hparams.connection_dropout,
                                    attention_dropout=hparams.attention_dropout,
                                    ) if hparams.num_slot_transformer_layers > 0 else None

    slot_predictor = torch.nn.Sequential(torch.nn.Linear(slot_decoder.output_dim, 1),
                                         torch.nn.Sigmoid(),
                                         )

    content_decoder = InsertionDecoder(input_dim=hparams.emb_sz,
                                       num_layers=hparams.num_dec_layers,
                                       num_heads=hparams.num_heads,
                                       feedforward_hidden_dim=hparams.emb_sz,
                                       feedforward_dropout=hparams.connection_dropout,
                                       residual_dropout=hparams.connection_dropout,
                                       attention_dropout=hparams.attention_dropout,

                                       # reuse the InsertionDecoder architecture for inputs other than slots
                                       concat_adjacent_repr_for_slots=False,
                                       )

    word_proj = MoSProjection(mixture_num=hparams.mixture_num,
                              input_dim=content_decoder.hidden_dim,
                              output_dim=vocab.get_vocab_size(),
                              flatten_softmax=False)

    dual_model = TransformerEncoder(input_dim=hparams.emb_sz,
                                    num_layers=hparams.num_dual_model_layers,
                                    feedforward_hidden_dim=hparams.emb_sz,
                                    feedforward_dropout=hparams.connection_dropout,
                                    residual_dropout=hparams.connection_dropout,
                                    attention_dropout=hparams.attention_dropout,
                                    ) if hparams.num_dual_model_layers > 0 else None

    model = DecoupledInsertionTransformer(vocab=vocab,
                                          encoder=encoder,
                                          slot_decoder=slot_decoder,
                                          slot_predictor=slot_predictor,
                                          content_decoder=content_decoder,
                                          word_predictor=word_proj,
                                          src_embedding=embedding,
                                          tgt_embedding=embedding,
                                          use_bleu=True,
                                          target_namespace="tokens",
                                          start_symbol=KeywordSPMInsTranslator.START_SYMBOL,
                                          eos_symbol=KeywordSPMInsTranslator.END_SYMBOL,
                                          mask_symbol=KeywordSPMInsTranslator.MASK_SYMBOL,
                                          max_decoding_step=hparams.max_decoding_len,
                                          span_end_threshold=hparams.span_end_threshold,
                                          stammering_window=hparams.stammering_window,
                                          alpha1=hparams.alpha1,
                                          alpha2=hparams.alpha2,
                                          topk=hparams.topk_words,
                                          slot_trans=slot_trans,
                                          dual_model=dual_model,
                                          )

    return model

class InsTrainingUpdater(TrainingUpdater):
    def __init__(self, *args, **kwargs):
        super(InsTrainingUpdater, self).__init__(*args, **kwargs)
        self._transform = None

    @classmethod
    def from_bot(cls, bot: TrialBot):
        updater = super().from_bot(bot)

        maskid = bot.vocab.get_token_index(KeywordSPMInsTranslator.MASK_SYMBOL)
        updater._transform = DecoupledInsTrans(maskid)
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
        training_targets = next(self._transform(tkwds, tgts, skill=(iterator.epoch + 1) / 10.))

        if device >= 0:
            srcs = move_to_device(srcs, device)
            training_targets = move_to_device(training_targets, device)

        output = model(srcs, training_targets=training_targets)

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

        output = model(srcs, inference_input=tkwds, references=tgts)
        return output

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

    bot = TrialBot(trial_name="decoupled_ins", get_model_func=get_model, args=args)
    if args.test:
        import trialbot
        new_engine = trialbot.training.trial_bot.Engine()
        new_engine.register_events(*Events)
        bot._engine = new_engine

        def predicted_output(bot: TrialBot):
            import json
            model = bot.model
            output = bot.state.output
            if output is None:
                return

            output = model.decode(output)
            def decode(output: List):
                elem = output[0]
                if isinstance(elem, str):
                    return bot.translator.spm.DecodePieces(output)
                elif isinstance(elem, list):
                    return [decode(x) for x in output]

            output["predicted_tokens"] = decode(output["predicted_tokens"])
            print(json.dumps(output["predicted_tokens"]))

        @bot.attach_extension(Events.ITERATION_COMPLETED)
        def predicted_char_output(bot: TrialBot):
            import json
            model = bot.model
            output = bot.state.output
            if output is None:
                return

            output = model.decode(output)
            def decode(output: List):
                elem = output[0]
                if isinstance(elem, str):
                    return "".join(output)
                elif isinstance(elem, list):
                    return [decode(x) for x in output]

            output["predicted_tokens"] = decode(output["predicted_tokens"])
            for x in output["predicted_tokens"]:
                print(x)

        @bot.attach_extension(Events.COMPLETED)
        def precomputed_bleu(bot: TrialBot):
            model: DecoupledInsertionTransformer = bot.model
            metrics = model.get_metrics(reset=False)
            print(metrics)

        bot.updater = InsTestingUpdater.from_bot(bot)
    else:
        from trialbot.training.extensions import every_epoch_model_saver

        def output_inspect(bot: TrialBot, keys):
            iteration = bot.state.iteration
            if iteration % 4 != 0:
                return

            output = bot.state.output
            bot.logger.info(", ".join(f"{k}={v}" for k, v in zip(keys, map(output.get, keys))))

        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)
        bot.add_event_handler(Events.ITERATION_COMPLETED, output_inspect, 100,
                              keys=["loss_slot_dec", "loss_cont_dec"])
        bot.updater = InsTrainingUpdater.from_bot(bot)
    bot.run()

if __name__ == '__main__':
    main()
