import sys
sys.path.insert(0, '..')
from typing import List, Generator, Tuple, Mapping, Optional
import logging
import os.path
import torch.nn
from torch.nn import functional as F
import itertools
import functools
import numpy as np
import random
import re
import copy

from trialbot.data import NSVocabulary, PADDING_TOKEN, Translator, RandomIterator
from trialbot.training import Registry, TrialBot, Events
from trialbot.training.updater import Updater
from trialbot.training.hparamset import HyperParamSet
from trialbot.utils.move_to_device import move_to_device
from utils.inner_loop_optimizers import LSLRGradientDescentLearningRule
from datasets.cached_retriever import IDCacheRetriever, HypIDCacheRetriever
from fairseq.optim.adafactor import Adafactor

from utils.root_finder import find_root
from utils.itertools import zip_cycle
_ROOT = find_root()

def _atis_base():
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
    p.weight_decay = 0.2
    p.char_emb_sz = 128
    p.char_hid_sz = 128
    p.dropout = .2
    p.discrete_dropout = .1

    p.TRAINING_LIMIT = 20
    p.batch_sz = 64
    # p.task_batch_sz = 8
    p.num_inner_loops = 3
    return p

def _django_base():
    p = _atis_base()
    p.TRAINING_LIMIT = 5
    return p

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

def get_aux_model(hparams, vocab):
    from experiments.build_model import get_re2_char_model, get_char_giant
    model: torch.nn.Module = get_char_giant(hparams, vocab)
    dev = torch.device('cpu') if hparams.DEVICE < 0 else torch.device(hparams.DEVICE)
    LSLRGD = LSLRGradientDescentLearningRule
    lslrgd: LSLRGD = LSLRGD(device=dev,
                            total_num_inner_loop_steps=hparams.num_inner_loops,
                            use_learnable_learning_rates=True,)
    lslrgd.initialise(dict(model.named_parameters()))
    return [model, lslrgd]
    # return torch.nn.ModuleDict({"model": model, "inner_optim": lslrgd})

def get_clean_model(hparams, vocab):
    model, lslrgd = get_aux_model(hparams, vocab)
    if hasattr(model, 'loss_weighting'):
        del model.loss_weighting
    return [model, lslrgd]

class MAMLUpdater(Updater):
    def __init__(self, models, iterators, optims,
                 retriever,
                 device,
                 grad_clip_val,
                 get_data_iter_fn,
                 num_inner_loop,
                 is_training: bool = True,
                 ):
        super().__init__(models, iterators, optims, device)
        self._retriever: IDCacheRetriever = retriever
        self._grad_clip_val = grad_clip_val
        self._get_data_iter = get_data_iter_fn
        self._num_inner_loop = num_inner_loop

        self.training = is_training

    def get_pseudo_task_batch(self, batch, batch_source: Optional[str] = None):
        """
        Use a batch of data
        :param batch: a batch of data
        :param batch_source: "train", "test", "dev", indicating the dataset from which the current batch comes
        :return:
        """
        if batch_source is None:
            batch_source = "train" if self.training else "test"

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

        device = self._device
        if device >= 0:
            sent_a = move_to_device(sent_a, device)
            sent_b = move_to_device(sent_b, device)
            sent_char_a = move_to_device(sent_char_a, device)
            sent_char_b = move_to_device(sent_char_b, device)
            label = move_to_device(label, device)

        return sent_a, sent_b, sent_char_a, sent_char_b, label

    def update_epoch(self):
        iterator = self._iterators[0]
        if iterator.is_new_epoch:
            self.stop_epoch()

        model, lslrgd = self._models
        if self.training:
            model.train()
            lslrgd.train()
        else:
            model.eval()
            lslrgd.eval()

        batch = next(iterator)
        if self.training:
            return self._trainer_update(batch, model, lslrgd)
        else:
            return self._evaluation_updater(batch, model, lslrgd)

    def _trainer_update(self, batch, model, inner_opt):

        init_params = copy.deepcopy(model.state_dict())
        task_losses = []

        all_task_data = self.get_pseudo_task_batch(batch, "train")
        task_iter = self._get_data_iter(all_task_data)

        for step in range(self._num_inner_loop):
            # each batch contains one example of every pseudo-tasks
            support_batch = next(task_iter)

            sent_a, sent_b, char_a, char_b, label = self.read_model_input(support_batch)
            model.zero_grad()

            loss_step = model(sent_a, sent_b, char_a, char_b, label)
            grad_params = dict(model.named_parameters())
            grads = torch.autograd.grad(loss_step, grad_params.values())
            torch.nn.utils.clip_grad_value_(grads, self._grad_clip_val)
            names_grads_wrt_params = dict(zip(grad_params.keys(), grads))

            new_weights = inner_opt.update_params(names_weights_dict=grad_params,
                                                  names_grads_wrt_params_dict=names_grads_wrt_params,
                                                  num_step=step)

            model.load_state_dict(new_weights, strict=False)
            model.zero_grad()
            query_batch = next(task_iter)
            sent_a, sent_b, char_a, char_b, label = self.read_model_input(query_batch)
            loss_eval_step = model(sent_a, sent_b, char_a, char_b, label)
            task_losses.append(loss_eval_step)

        model.load_state_dict(init_params)
        outer_opts = self._optims   # two optims for model and lslrgd respectively
        for opt in outer_opts:
            opt.zero_grad()

        if len(task_losses) == 0:
            return {"loss": 0, "task_loss_count": len(task_losses), "msg": "all task has empty support"}

        loss_total = sum(task_losses) / len(task_losses)
        loss_total.backward()
        for opt in outer_opts:
            opt.step()

        return {"loss": loss_total, "task_loss_count": len(task_losses)}

    def _evaluation_updater(self, batch, model, inner_opt):

        all_task_data = self.get_pseudo_task_batch(batch, "train")
        task_iter = self._get_data_iter(all_task_data)

        for step in range(self._num_inner_loop):
            # each batch contains one example of every pseudo-tasks
            support_batch = next(task_iter)

            sent_a, sent_b, char_a, char_b, label = self.read_model_input(support_batch)
            with torch.enable_grad():
                model.train()
                model.zero_grad()
                loss_step = model(sent_a, sent_b, char_a, char_b, label)
                grad_params = dict(model.named_parameters())
                grads = torch.autograd.grad(loss_step, grad_params.values())

            torch.nn.utils.clip_grad_value_(grads, self._grad_clip_val)
            names_grads_wrt_params = dict(zip(grad_params.keys(), grads))

            new_weights = inner_opt.update_params(names_weights_dict=grad_params,
                                                  names_grads_wrt_params_dict=names_grads_wrt_params,
                                                  num_step=step)

            model.load_state_dict(new_weights, strict=False)

        model.eval()
        sent_a, sent_b, char_a, char_b, label = self.read_model_input(batch)
        rankings = model.inference(sent_a, sent_b, char_a, char_b)
        # task_logits.append(model.inference(sent_a, sent_b, char_a, char_b))
        output = model.forward_loss_weight(*rankings)

        # output = torch.cat(scores, dim=0)
        # output = torch.cat(task_logits, dim=0).log_softmax(dim=-1)
        # correct_score = output[:, 1]

        return {"prediction": output, "ranking_score": output,
                "ex_id": batch["ex_id"], "hyp_rank": batch["hyp_rank"]}


    @classmethod
    def from_bot(cls, bot: TrialBot):
        args, hparams, model_closure = bot.args, bot.hparams, bot.models
        model, lslrgd = model_closure
        device = args.device

        # We treat inner loop and outer loop different and choose different optimizers for each.
        outer_optim = Adafactor(model.parameters(), weight_decay=hparams.weight_decay)
        bot.logger.info("Use Adafactor as outer optimizer: " + str(outer_optim))
        outer_optim_for_inner_optim = Adafactor(lslrgd.parameters(), weight_decay=hparams.weight_decay)
        bot.logger.info("Use Adafactor as outer optimizer: " + str(outer_optim_for_inner_optim))

        from functools import partial

        # either testing or debugging training, iter should be static, otherwise iterator returns data dynamically.
        # a dynamic iterator enables data shuffling and repeats several epochs.
        is_dynamic_iter = not (args.test or args.debug)
        dataset = bot.test_set if args.test else bot.train_set
        iterator = RandomIterator(dataset, bot.hparams.batch_sz, bot.translator,
                                  shuffle=is_dynamic_iter, repeat=is_dynamic_iter)
        if args.debug and args.skip:
            iterator.reset(args.skip)

        support_set_iter_fn = partial(RandomIterator, shuffle=True, repeat=True,
                                      batch_size=hparams.batch_sz, translator=bot.translator,)
        if hasattr(model, 'loss_weighting'):
            del model.loss_weighting

        # NL similarity uses only example id as key, LF similarity requires hyp id to denote a concrete example
        retriever_cls = IDCacheRetriever if '_nl_' in args.hparamset else HypIDCacheRetriever
        retriever = retriever_cls(filename=hparams.retriever_index_path, dataset=bot.train_set)
        updater = cls(model_closure, iterator,
                      [outer_optim, outer_optim_for_inner_optim],
                      retriever, device, hparams.GRAD_CLIPPING,
                      support_set_iter_fn, hparams.num_inner_loops, is_training=(not args.test))
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

    if args.test:
        import trialbot
        bot = TrialBot(trial_name="meta_ranker", get_model_func=get_clean_model, args=args)
        bot.updater = MAMLUpdater.from_bot(bot)
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

            output_keys = ("ex_id", "hyp_rank", "ranking_score")
            for eid, hyp_rank, score in zip(*map(output.get, output_keys)):
                print(json.dumps(dict(zip(output_keys, (eid, hyp_rank, score.item())))))

    else:
        from utils.trial_bot_extensions import save_multiple_models_every_num_iters, save_multiple_models_per_epoch
        bot = TrialBot(trial_name="meta_ranker", get_model_func=get_aux_model, args=args)
        bot.updater = MAMLUpdater.from_bot(bot)
        bot.translator.turn_special_token(on=True)

        @bot.attach_extension(Events.ITERATION_COMPLETED)
        def end_with_nan_loss(bot: TrialBot):
            output = bot.state.output
            if output is None:
                return
            loss = output["loss"]
            def _isnan(x):
                if isinstance(x, torch.Tensor):
                    return bool(torch.isnan(x).any())
                elif isinstance(x, np.ndarray):
                    return bool(np.isnan(x).any())
                else:
                    import math
                    return math.isnan(x)

            if _isnan(loss):
                bot.logger.error("NaN loss encountered, training ended")
                bot.state.epoch = bot.hparams.TRAINING_LIMIT + 1
                bot.updater.stop_epoch()

        def output_inspect(bot: TrialBot, keys):
            iteration = bot.state.iteration
            output = bot.state.output
            if iteration % 4 != 0 or output is None:
                return
            bot.logger.info(", ".join(f"{k}={v}" for k, v in zip(keys, map(output.get, keys))))

        bot.add_event_handler(Events.EPOCH_COMPLETED, save_multiple_models_per_epoch, 100)
        bot.add_event_handler(Events.ITERATION_COMPLETED, output_inspect, 100, keys=["loss", "task_loss_count"])
    bot.run()

if __name__ == '__main__':
    main()


