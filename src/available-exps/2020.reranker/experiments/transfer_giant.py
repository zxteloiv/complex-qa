import sys
sys.path.insert(0, '..')
from typing import Optional
import logging
import torch.nn
from fairseq.optim.adafactor import Adafactor

import trialbot
from trialbot.training import Registry, TrialBot, Events
from trialbot.training.hparamset import HyperParamSet
from trialbot.data import Iterator, RandomIterator
from trialbot.utils.move_to_device import move_to_device
from trialbot.training.updater import Updater
from datasets.cached_retriever import IDCacheRetriever, HypIDCacheRetriever
import functools
import copy
import os.path

from utils.itertools import zip_cycle
from utils.root_finder import find_root
_ROOT = find_root()

def _atis_base():
    """base atis config, using deep giant"""
    p = HyperParamSet.common_settings(_ROOT)
    p.alignment = "bilinear"    # identity, linear, bilinear
    p.prediction = "full"     # simple, full, symmetric
    p.encoder = "bilstm"
    p.pooling = "neo"  # neo or vanilla
    p.fusion = "neo"    # neo or vanilla
    p.connection = "aug"      # none, residual, aug
    p.num_classes = 2     # either 0 (true) or 1 (false), only 2 classes
    p.emb_sz = 256
    p.hidden_size = 256
    p.num_stacked_block = 4
    p.num_stacked_encoder = 1
    p.dropout = .2
    p.discrete_dropout = .1
    p.TRAINING_LIMIT = 200
    p.weight_decay = 0.2
    p.batch_sz = 32
    p.char_emb_sz = 128
    p.char_hid_sz = 128
    p.num_inner_loops = 4
    return p

def _django_base():
    hparams = _atis_base()
    hparams.TRAINING_LIMIT = 60
    return hparams

@Registry.hparamset()
def atis_nl_ngram():
    p = _atis_base()
    p.retriever_index_path = os.path.join(_ROOT, 'data', '_similarity_index', 'atis_nl_ngram.bin')
    return p

@Registry.hparamset()
def atis_nl_bert():
    p = _atis_base()
    p.retriever_index_path = os.path.join(_ROOT, 'data', '_similarity_index', 'atis_bert_nl.bin')
    return p

@Registry.hparamset()
def atis_lf_ngram():
    p = _atis_base()
    p.retriever_index_path = os.path.join(_ROOT, 'data', '_similarity_index', 'atis_lf_ngram.bin')
    return p

@Registry.hparamset()
def atis_lf_ted():
    p = _atis_base()
    p.retriever_index_path = os.path.join(_ROOT, 'data', '_similarity_index', 'atis_lf_ted.bin')
    return p

@Registry.hparamset()
def django_nl_ngram():
    p = _django_base()
    p.retriever_index_path = os.path.join(_ROOT, 'data', '_similarity_index', 'django_nl_ngram.bin')
    return p

@Registry.hparamset()
def django_nl_bert():
    p = _django_base()
    p.retriever_index_path = os.path.join(_ROOT, 'data', '_similarity_index', 'django_bert_nl.bin')
    return p

@Registry.hparamset()
def django_lf_ngram():
    p = _django_base()
    p.retriever_index_path = os.path.join(_ROOT, 'data', '_similarity_index', 'django_lf_ngram.bin')
    return p

@Registry.hparamset()
def django_lf_ted():
    p = _django_base()
    p.retriever_index_path = os.path.join(_ROOT, 'data', '_similarity_index', 'django_lf_ted.bin')
    return p

import datasets.atis_rank
import datasets.atis_rank_translator
import datasets.django_rank
import datasets.django_rank_translator

def get_model(hparams, vocab):
    from experiments.build_model import get_char_giant
    return get_char_giant(hparams, vocab)

class TransferUpdater(Updater):
    def __init__(self, models, iterators, optims,
                 retriever,
                 device,
                 get_data_iter_fn,
                 num_inner_loop,
                 batch_source: str = "test",
                 ):
        super().__init__(models, iterators, optims, device)
        self._retriever: IDCacheRetriever = retriever
        self._get_data_iter = get_data_iter_fn
        self._num_inner_loop = num_inner_loop
        self.batch_source = batch_source

    def get_pseudo_task_batch(self, batch, batch_source: Optional[str] = None):
        """
        Use a batch of data
        :param batch: a batch of data
        :param batch_source: "train", "test", "dev", indicating the dataset from which the current batch comes
        :return:
        """
        batch_source = batch_source or self.batch_source

        # each element is a list of similar examples,
        # corresponding to an example which indicates a different pseudotask
        # the None or empty lists (i.e., no similar examples found for some pseudotask) are filtered.
        all_similar_examples = filter(None, map(lambda e: self._retriever.search(e, batch_source), batch['_raw']))

        # the support set is a concatenated list of all the pseudo-tasks.
        #
        support_sets = functools.reduce(lambda x, y: list(x) + list(y), zip_cycle(*all_similar_examples), [])
        return support_sets

    def read_model_input(self, batch):
        sent_a = batch['source_tokens']
        sent_b = batch['hyp_tokens']
        sent_char_a = batch['src_char_ids']
        sent_char_b = batch['hyp_char_ids']
        label = batch['hyp_label']
        rank = batch['hyp_rank']

        device = self._device
        if device >= 0:
            sent_a = move_to_device(sent_a, device)
            sent_b = move_to_device(sent_b, device)
            sent_char_a = move_to_device(sent_char_a, device)
            sent_char_b = move_to_device(sent_char_b, device)
            label = move_to_device(label, device)
            rank = move_to_device(rank, device)

        return sent_a, sent_b, sent_char_a, sent_char_b, label, rank

    def update_epoch(self):
        iterator = self._iterators[0]
        model = self._models[0]
        optim = self._optims[0]
        if iterator.is_new_epoch:
            self.stop_epoch()

        model.eval()
        batch = next(iterator)
        support_data = self.get_pseudo_task_batch(batch)
        task_iter = self._get_data_iter(support_data)
        init_params = copy.deepcopy(model.state_dict())

        with torch.enable_grad():
            model.train()
            for step, support_batch in enumerate(task_iter):
                if step >= self._num_inner_loop:
                    break
                sent_a, sent_b, char_a, char_b, label, rank = self.read_model_input(support_batch)
                model.zero_grad()
                loss_step = model(sent_a, sent_b, char_a, char_b, label, rank)
                loss_step.backward()
                optim.step()

        model.eval()
        sent_a, sent_b, char_a, char_b, label, rank = self.read_model_input(batch)
        rankings = model.inference(sent_a, sent_b, char_a, char_b, rank)
        # task_logits.append(model.inference(sent_a, sent_b, char_a, char_b))
        output = model.forward_loss_weight(*rankings)

        model.load_state_dict(init_params)  # reset the original model

        return {"ranking_score": output,
                "rank_match": rankings[0],
                "rank_a2b": rankings[1],
                "rank_b2a": rankings[2],
                "label": label,
                "ex_id": batch["ex_id"], "hyp_rank": batch["hyp_rank"]}


    @classmethod
    def from_bot(cls, bot: TrialBot):
        args, hparams, model = bot.args, bot.hparams, bot.model
        device = args.device

        # We treat inner loop and outer loop different and choose different optimizers for each.
        optim = Adafactor(model.parameters(), weight_decay=hparams.weight_decay)
        bot.logger.info("Use Adafactor as outer optimizer: " + str(optim))
        from functools import partial

        # either testing or debugging training, iter should be static, otherwise iterator returns data dynamically.
        # a dynamic iterator enables data shuffling and repeats several epochs.
        is_dynamic_iter = not (args.test or args.debug)
        if args.test and args.dev:
            dataset = bot.dev_set
            batch_source = "dev"
        elif args.test:
            dataset = bot.test_set
            batch_source = "test"
        else:
            dataset = bot.train_set
            batch_source = "train"

        iterator = RandomIterator(dataset, hparams.batch_sz, bot.translator, shuffle=is_dynamic_iter, repeat=is_dynamic_iter)
        if args.debug and args.skip:
            iterator.reset(args.skip)

        support_set_iter_fn = partial(RandomIterator, shuffle=True, repeat=True,
                                      batch_size=hparams.batch_sz, translator=bot.translator,)

        # NL similarity uses only example id as key, LF similarity requires hyp id to denote a concrete example
        retriever_cls = IDCacheRetriever if '_nl_' in args.hparamset else HypIDCacheRetriever
        retriever = retriever_cls(filename=hparams.retriever_index_path, dataset=bot.train_set)
        updater = cls(model, iterator, optim, retriever, device, support_set_iter_fn,
                      hparams.num_inner_loops, batch_source)
        return updater

def main():
    import sys
    args = sys.argv[1:]
    args += ['--seed', '2020']
    if '--dataset' not in sys.argv:
        args += ['--dataset', 'atis_five_hyp']
    if '--translator' not in sys.argv:
        args += ['--translator', 'atis_rank']

    parser = TrialBot.get_default_parser()
    parser.add_argument('--dev', action='store_true', help="use dev data for testing mode, only works with --test opt")
    args = parser.parse_args(args)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.INFO)

    if hasattr(args, "seed") and args.seed:
        from utils.fix_seed import fix_seed
        logging.info(f"set seed={args.seed}")
        fix_seed(args.seed)

    bot = TrialBot(trial_name="transfer_giant", get_model_func=get_model, args=args)

    bot.translator.turn_special_token(on=True)
    new_engine = trialbot.training.trial_bot.Engine()
    new_engine.register_events(*Events)
    bot._engine = new_engine

    @bot.attach_extension(Events.ITERATION_COMPLETED)
    def print_output(bot: TrialBot):
        import json
        output = bot.state.output
        if output is None:
            return

        output_keys = ("ex_id", "hyp_rank", "ranking_score", "rank_match", "rank_a2b", "rank_b2a")
        for eid, hyp_rank, score, r_m, r_a2b, r_b2a in zip(*map(output.get, output_keys)):
            print(json.dumps(dict(zip(output_keys, (eid, hyp_rank.item(), score.item(),
                                                    r_m.item(), r_a2b.item(), r_b2a.item())))))

    bot.updater = TransferUpdater.from_bot(bot)
    bot.run()

if __name__ == '__main__':
    main()


