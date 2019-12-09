import sys
sys.path.insert(0, '..')
from typing import List, Generator, Tuple, Mapping, Optional
import logging
import os.path
import torch.nn
from torch.nn import functional as F
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
from datasets.cached_retriever import IDCacheRetriever, HypIDCacheRetriever, SimRetriever

from utils.root_finder import find_root
_ROOT = find_root()

@Registry.hparamset()
def atis_five_lstm():
    hparams = HyperParamSet.common_settings(_ROOT)
    hparams.emb_sz = 300
    hparams.hidden_size = 150
    hparams.encoder_kernel_size = 3
    hparams.num_classes = 2     # either 0 (true) or 1 (false), only 2 classes
    hparams.num_stacked_block = 3
    hparams.num_stacked_encoder = 2
    hparams.dropout = .5
    hparams.fusion = "full"         # simple, full
    hparams.alignment = "linear"    # identity, linear
    hparams.connection = "aug"      # none, residual, aug
    hparams.prediction = "full"     # simple, full, symmetric

    hparams.encoder = "lstm"
    hparams.TRAINING_LIMIT = 20
    hparams.batch_sz = 16
    hparams.num_inner_loops = 3
    hparams.disable_inner_loop = False
    hparams.task_batch_sz = 8
    hparams.retriever_index_path = os.path.join(_ROOT, 'data', '_similarity_index', 'atis_bert_nl.bin')
    return hparams

@Registry.hparamset()
def atis_five_v2():
    p = atis_five_lstm()
    p.emb_sz = 256
    p.dropout = .2
    p.hidden_size = 128
    p.encoder = "bilstm"
    p.batch_sz = 16
    p.num_inner_loops = 3
    p.task_batch_sz = 16
    p.TRAINING_LIMIT = 5
    p.connection = "aug"
    return p

@Registry.hparamset()
def atis_five_v2_no_dropout():
    p = atis_five_v2()
    p.dropout = .0
    return p

@Registry.hparamset()
def atis_five_v3():
    p = atis_five_v2()
    p.retriever_index_path = (
        os.path.join(_ROOT, 'data', '_similarity_index', 'atis_ted_lf_train.bin'),
        os.path.join(_ROOT, 'data', '_similarity_index', 'atis_ted_lf_dev.bin'),
        os.path.join(_ROOT, 'data', '_similarity_index', 'atis_ted_lf_test.bin'),
    )
    return p

@Registry.hparamset()
def atis_five_v3_no_dropout():
    p = atis_five_v3()
    p.dropout = .0
    return p

@Registry.hparamset()
def atis_five_v4():
    p2 = atis_five_v2()
    p3 = atis_five_v3()
    p3.retriever_index_path = (
        p2.retriever_index_path,
        p3.retriever_index_path,
    )  # Tuple[str, Tuple[str, str, str]]

    p3.emb_sz = 256
    p3.hidden_size = 128
    p3.num_stacked_block = 2
    p3.num_stacked_encoder = 2
    p3.dropout = .2
    p3.TRAINING_LIMIT = 5
    p3.batch_sz = 24
    p3.num_inner_loops = 3
    p3.task_batch_sz = 16
    return p3

@Registry.hparamset()
def atis_five_testing_no_dropout():
    hparams = atis_five_lstm()
    hparams.dropout = 0.
    return hparams

@Registry.hparamset()
def atis_five_crude_testing():
    hparams = atis_five_lstm()
    hparams.disable_inner_loop = True
    return hparams

@Registry.hparamset()
def django_fifteen():
    hparams = atis_five_lstm()
    hparams.TRAINING_LIMIT = 5
    hparams.retriever_index_path = os.path.join(_ROOT, 'data', '_similarity_index', 'django_bert_nl.bin')
    return hparams

@Registry.hparamset()
def django_fifteen_crude_testing():
    hparams = django_fifteen()
    hparams.disable_inner_loop = True
    return hparams

@Registry.hparamset()
def django_fifteen_v2():
    p = atis_five_v2()
    p.retriever_index_path = os.path.join(_ROOT, 'data', '_similarity_index', 'django_bert_nl.bin')
    p.emb_sz = 256
    p.hidden_size = 128
    p.num_stacked_block = 2
    p.num_stacked_encoder = 2
    p.dropout = .2
    p.TRAINING_LIMIT = 3
    p.batch_sz = 24
    p.num_inner_loops = 3
    p.task_batch_sz = 16
    return p

import datasets.atis_rank
import datasets.atis_rank_translator
import datasets.django_rank
import datasets.django_rank_translator

def get_aux_model(hparams, vocab):
    from experiments.build_model import get_re2_variant
    model: torch.nn.Module = get_re2_variant(hparams, vocab)
    dev = torch.device('cpu') if hparams.DEVICE < 0 else torch.device(hparams.DEVICE)
    LSLRGD = LSLRGradientDescentLearningRule
    lslrgd: LSLRGD = LSLRGD(device=dev,
                            total_num_inner_loop_steps=hparams.num_inner_loops,
                            use_learnable_learning_rates=True,)
    lslrgd.initialise(dict(model.named_parameters()))
    return torch.nn.ModuleDict({"model": model, "inner_optim": lslrgd})

class MetaRankerTrainingUpdater(Updater):
    def __init__(self, models, iterators, optims,
                 schedulers,
                 retriever,
                 device,
                 grad_clip_val,
                 get_data_iter_fn,
                 num_inner_loop,
                 ):
        super().__init__(models, iterators, optims, device)
        self._retriever: IDCacheRetriever = retriever
        self._scheds = schedulers if isinstance(schedulers, list) else [schedulers]
        self._grad_clip_val = grad_clip_val
        self._get_data_iter = get_data_iter_fn
        self._num_inner_loop = num_inner_loop

    def stop_epoch(self):
        super().stop_epoch()
        outer_sche: torch.optim.lr_scheduler.CosineAnnealingLR = self._scheds[0]
        outer_sche.step()

    def update_epoch(self):
        model_closure, iterator = self._models[0], self._iterators[0]
        if iterator.is_new_epoch:
            self.stop_epoch()

        device = self._device
        model_closure.train()
        model: torch.nn.Module
        inner_opt: LSLRGradientDescentLearningRule
        model, inner_opt = model_closure['model'], model_closure['inner_optim']
        outer_opt = self._optims[0]

        def read_model_input(batch):
            sent_a, sent_b, label = list(map(batch.get, ("source_tokens", "hyp_tokens", "hyp_label")))
            if device >= 0:
                sent_a = move_to_device(sent_a, device)
                sent_b = move_to_device(sent_b, device)
                label = move_to_device(label, device)
            return sent_a, sent_b, label

        batch = next(iterator)
        init_params = copy.deepcopy(model.state_dict())
        task_losses = []
        for task_data in self.pseudo_task_data_generator(batch, "train"):

            model.load_state_dict(init_params)
            task_iter = self._get_data_iter(task_data)
            # run multi-step
            for step in range(self._num_inner_loop):
                support_batch = next(task_iter)
                sent_a, sent_b, label = read_model_input(support_batch)
                model.zero_grad()

                loss_step = F.cross_entropy(model(sent_a, sent_b), label)
                grad_params = dict(model.named_parameters())
                grads = torch.autograd.grad(loss_step, grad_params.values())
                torch.nn.utils.clip_grad_value_(grads, self._grad_clip_val)
                names_grads_wrt_params = dict(zip(grad_params.keys(), grads))

                new_weights = inner_opt.update_params(names_weights_dict=grad_params,
                                                      names_grads_wrt_params_dict=names_grads_wrt_params,
                                                      num_step=step)
                model.load_state_dict(new_weights)
                model.zero_grad()
                query_batch = next(task_iter)
                sent_a, sent_b, label = read_model_input(query_batch)
                loss_eval_step = F.cross_entropy(model(sent_a, sent_b), label)
                task_losses.append(loss_eval_step)

        model.load_state_dict(init_params)
        outer_opt.zero_grad()
        loss_total = sum(task_losses) / len(task_losses)
        loss_total.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), self._grad_clip_val)
        outer_opt.step()

        return {"loss": loss_total, "task_loss_count": len(task_losses)}

    def pseudo_task_data_generator(self, batch, batch_source: str = "train"):
        # every example is a pseudo task
        for raw_example in batch['_raw']:
            # list of similar examples
            similar_examples: list = self._retriever.search(raw_example, batch_source)
            yield similar_examples

    @classmethod
    def from_bot(cls, bot: TrialBot):
        args, hparams, model = bot.args, bot.hparams, bot.model

        # We treat inner loop and outer loop different and choose different optimizers for each.
        outer_optim = torch.optim.AdamW(model.parameters(), hparams.ADAM_LR, hparams.ADAM_BETAS)
        bot.logger.info("Outer optim.: " + re.sub('  +', ', ', str(outer_optim).replace('\n', '')))
        outer_sche = torch.optim.lr_scheduler.CosineAnnealingLR(outer_optim, hparams.TRAINING_LIMIT, hparams.ADAM_LR)
        bot.logger.info("Learning scheduler: " + re.sub('  +', ', ', str(outer_optim).replace('\n', '')))

        device = args.device
        repeat_iter = not args.debug
        shuffle_iter = not args.debug
        iterator = RandomIterator(bot.train_set, bot.hparams.batch_sz, bot.translator,
                                  shuffle=shuffle_iter, repeat=repeat_iter)
        if args.debug and args.skip:
            iterator.reset(args.skip)

        if isinstance(hparams.retriever_index_path, str):
            retriever = IDCacheRetriever(filename=hparams.retriever_index_path, dataset=bot.train_set)
        elif len(hparams.retriever_index_path) == 2:
            rpath = hparams.retriever_index_path
            retriever = SimRetriever(nlfile=rpath[0], lffiles=rpath[1], dataset=bot.train_set)
        elif len(hparams.retriever_index_path) == 3:
            retriever = HypIDCacheRetriever(filenames=hparams.retriever_index_path, dataset=bot.train_set)
        else:
            raise ValueError('failed to init retriever')

        from functools import partial
        updater = cls(model, iterator, outer_optim, outer_sche,
                      retriever, device, hparams.GRAD_CLIPPING,
                      partial(RandomIterator, batch_size=hparams.task_batch_sz, translator=bot.translator,
                              shuffle=True, repeat=True),
                      hparams.num_inner_loops)
        return updater

class MetaRankerTestingUpdater(Updater):
    def __init__(self, models, iterators,
                 retriever,
                 device,
                 grad_clip_val,
                 get_data_iter_fn,
                 num_inner_loop,
                 ):
        super().__init__(models, iterators, [None], device)
        self._retriever: IDCacheRetriever = retriever
        self._grad_clip_val = grad_clip_val
        self._get_data_iter = get_data_iter_fn
        self._num_inner_loop = num_inner_loop

    def update_epoch(self):
        model_closure, iterator = self._models[0], self._iterators[0]
        if iterator.is_new_epoch:
            self.stop_epoch()

        device = self._device
        model_closure.eval()
        model: torch.nn.Module
        inner_opt: LSLRGradientDescentLearningRule
        model, inner_opt = model_closure['model'], model_closure['inner_optim']

        def read_model_input(batch):
            sent_a, sent_b, label = list(map(batch.get, ("source_tokens", "hyp_tokens", "hyp_label")))
            if device >= 0:
                sent_a = move_to_device(sent_a, device)
                sent_b = move_to_device(sent_b, device)
                label = move_to_device(label, device)
            return sent_a, sent_b, label

        batch = next(iterator)
        init_params = copy.deepcopy(model.state_dict())
        task_logits = []
        for query_example, task_data in zip(batch['_raw'], self.pseudo_task_data_generator(batch, "test")):
            model.load_state_dict(init_params)
            task_iter: RandomIterator = self._get_data_iter(task_data)
            # run multi-step
            for step in range(self._num_inner_loop):
                support_batch = next(task_iter)
                sent_a, sent_b, label = read_model_input(support_batch)
                with torch.enable_grad():
                    model.train()
                    model.zero_grad()
                    loss_step = F.cross_entropy(model(sent_a, sent_b), label)
                    grad_params = dict(model.named_parameters())
                    grads = torch.autograd.grad(loss_step, grad_params.values())

                model.eval()
                torch.nn.utils.clip_grad_value_(grads, self._grad_clip_val)
                names_grads_wrt_params = dict(zip(grad_params.keys(), grads))
                new_weights = inner_opt.update_params(names_weights_dict=grad_params,
                                                      names_grads_wrt_params_dict=names_grads_wrt_params,
                                                      num_step=step)
                model.load_state_dict(new_weights)

            translator: Translator = task_iter.translator
            query_batch = translator.batch_tensor([translator.to_tensor(query_example)])
            sent_a, sent_b, label = read_model_input(query_batch)
            task_logits.append(model(sent_a, sent_b))

        output = torch.cat(task_logits, dim=0).log_softmax(dim=-1)
        correct_score = output[:, 1]

        return {"prediction": output, "ranking_score": correct_score,
                "ex_id": batch["ex_id"], "hyp_rank": batch["hyp_rank"]}

    def pseudo_task_data_generator(self, batch, batch_source: str = "test"):
        # every example is a pseudo task
        for raw_example in batch['_raw']:
            # list of similar examples
            similar_examples: list = self._retriever.search(raw_example, batch_source)
            yield similar_examples

    @classmethod
    def from_bot(cls, bot: TrialBot):
        args, hparams, model = bot.args, bot.hparams, bot.model
        device = args.device
        iterator = RandomIterator(bot.test_set, bot.hparams.batch_sz, bot.translator, shuffle=False, repeat=False)
        if isinstance(hparams.retriever_index_path, str):
            retriever = IDCacheRetriever(filename=hparams.retriever_index_path, dataset=bot.train_set)
        elif len(hparams.retriever_index_path) == 2:
            rpath = hparams.retriever_index_path
            retriever = SimRetriever(nlfile=rpath[0], lffiles=rpath[1], dataset=bot.train_set)
        elif len(hparams.retriever_index_path) == 3:
            retriever = HypIDCacheRetriever(filenames=hparams.retriever_index_path, dataset=bot.train_set)
        else:
            raise ValueError('failed to init retriever')

        from functools import partial
        updater = cls(model, iterator, retriever, device, hparams.GRAD_CLIPPING,
                      partial(RandomIterator,
                              batch_size=hparams.task_batch_sz,
                              translator=bot.translator,
                              shuffle=True, repeat=True),
                      hparams.num_inner_loops if not hparams.disable_inner_loop else 0)
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

    bot = TrialBot(trial_name="meta_ranker", get_model_func=get_aux_model, args=args)
    if args.test:
        import trialbot
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

        bot.updater = MetaRankerTestingUpdater.from_bot(bot)
    else:
        from trialbot.training.extensions import every_epoch_model_saver
        from utils.trial_bot_extensions import save_model_every_num_iters

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

        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)
        bot.add_event_handler(Events.ITERATION_COMPLETED, save_model_every_num_iters, 100, interval=25)
        bot.add_event_handler(Events.ITERATION_COMPLETED, output_inspect, 100, keys=["loss", "task_loss_count"])
        bot.updater = MetaRankerTrainingUpdater.from_bot(bot)
    bot.run()

if __name__ == '__main__':
    main()


