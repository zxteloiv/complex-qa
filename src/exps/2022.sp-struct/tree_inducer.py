import logging
import math
import os
import pickle
from collections import defaultdict
from functools import partial
from operator import itemgetter
from statistics import fmean, stdev
from typing import Tuple, Union, List, Dict, Any, Callable, Literal

import ot
import torch.nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from trialbot.data import RandomIterator
from trialbot.utils.root_finder import find_root
from trialbot.training.hparamset import HyperParamSet
from trialbot.training import Registry, TrialBot, Updater, Events
import os.path as osp
import sys

SRC_PATH = find_root('.SRC', osp.dirname(__file__))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from shujuji import install_semantic_parsing_datasets, get_field_names_by_prefix
from utils.trialbot.dummy_translator import install_dummy_translator
from utils.trialbot.setup_cli import setup as setup_cli
from utils.trialbot.setup_bot import setup_bot
from utils.select_optim import select_optim
from utils.tree import Tree, PreOrderTraverse, PostOrderTraverse
from utils.seq_collector import SeqCollector

from moving_metric_llm import MovingMetrics, Embedder


def main():
    install_semantic_parsing_datasets()
    install_dummy_translator(filter_none=True)
    if os.getenv('CUDA_VISIBLE_DEVICES') is None:
        raise OSError("Using accelerate package requires manually setting the ENV variable.")

    args = setup_cli(device=0, seed=2021, translator='dummy', hparamset='base')

    bot = TrialBot(args, 'tree_inducer', get_model)
    ds_field_names = get_field_names_by_prefix(args.dataset)
    bot.translator.gather_keys = ds_field_names
    # init additional components
    bot.embedder = get_moving_metric(bot)

    @bot.attach_extension(Events.STARTED)
    def print_llm(bot):
        bot.logger.info("LLM Spec:\n" + str(bot.embedder.model))
        bot.logger.info("Tokenizer Spec:\n" + str(bot.embedder.tok))

    if args.test:
        eval_inducer(args, bot)
    else:
        train_inducer(args, bot)


def train_inducer(args, bot: TrialBot):
    bot.updater = InducerTrainingUpdater.from_bot(bot)
    bot = setup_bot(bot, True, False, False, False, False, True)
    bot.run()


def eval_inducer(args, bot):
    bot.updater = InducerEvalUpdater.from_bot(bot)
    bot = setup_bot(bot, False, False, False, False, False, True)
    trees = defaultdict(list)
    ys = defaultdict(list)

    @bot.attach_extension(Events.ITERATION_COMPLETED)
    def collect_trees(bot: TrialBot):
        output = bot.state.output
        ds_id = bot.updater.ds_id
        trees[ds_id] += output['y_tree']
        ys[ds_id] += output['y']

    bot.updater.set_inducing_ds('train')
    bot.run()

    bot.updater.set_inducing_ds('test')
    bot.run()

    pickle.dump({"ys": ys, "trees": trees}, open('output.pkl', 'wb'))


def get_model(p, vocab):
    # get log prob(t): (num_toks, 1) <- (num_toks, llm_hid_sz)
    return torch.nn.Sequential(
        torch.nn.Linear(4096, p.hid_sz),
        torch.nn.Mish(),
        torch.nn.Linear(p.hid_sz, 1),
        torch.nn.Sigmoid()
    )


def get_moving_metric(bot):
    args, p = bot.args, bot.hparams
    embedder = Embedder(p.llm_path, device=args.device, max_tok_num=p.max_tok_num,
                        use_causal_lm_cls=p.use_causal_lm_cls,)
    return embedder


class InducerTrainingUpdater(Updater):
    @classmethod
    def from_bot(cls, bot: TrialBot) -> 'Updater':
        args, p = bot.args, bot.hparams
        iterators = list(map(lambda d: RandomIterator(len(d), p.batch_sz), bot.datasets))
        optim = select_optim(p, bot.model.parameters())
        updater = cls(bot.models, iterators, [optim], device=args.device)
        updater.bot = bot   # save the backward reference for convenience
        updater.num_turns = p.num_turns
        updater.num_k = p.num_tree_sampling
        updater.grad_clip = p.GRAD_CLIPPING
        updater.logger = logging.getLogger(cls.__name__)
        return updater

    def update_epoch(self):
        datasets = self.bot.datasets    # noqa

        # sample and embed data, which is fixed during each time of tree sampling
        train_turns, test_turns = [], []
        for turn in range(self.num_turns):
            train_samples = self.get_samples(datasets[0], self._iterators[0])
            train_turns.append(self.embed_data(*train_samples))

            if self._iterators[0].is_end_of_epoch:
                self.stop_epoch()

            test_samples = self.get_samples(datasets[-1], self._iterators[-1])
            test_turns.append(self.embed_data(*test_samples))

        self.logger.info('finished embedding the samples of every turn.')

        optim = self._optims[0]
        optim.zero_grad()
        losses = []
        for k in range(self.num_k):
            loss_k = self.once_tree_sampling(train_turns, test_turns)
            self.logger.info(f'conducted the {k}th tree sampling with loss={loss_k}')
            losses.append(loss_k)

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_value_(self.bot.model.parameters(), self.grad_clip)
        optim.step()
        return {'loss': sum(losses)}

    def get_samples(self, dataset, iterator):
        indices = next(iterator)
        translator = self.bot.translator
        samples = translator.batch_tensor([translator.to_tensor(dataset[i]) for i in indices])
        key_pairs = get_field_names_by_prefix(self.bot.args.dataset)
        return itemgetter(*key_pairs)(samples)

    def embed_data(self, x_data: List[str], y_data: List[str]) -> List[Dict[str, Any]]:
        list_of_dicts = []

        for x, y in zip(x_data, y_data):
            embedder: Embedder = self.bot.embedder    # noqa

            x_emb, x_id = embedder.embed(x)  # (num_toks, emb)
            x_emb = x_emb.float().clone().detach_()
            sx_emb = embedder.compute_flat_seq_emb(x_emb, x_id)

            y_emb, y_id = embedder.embed(y)  # (num_toks, emb)
            y_emb = y_emb.float().clone().detach_()

            list_of_dicts.append(dict(x=x, x_id=x_id.tolist(), x_emb=x_emb, sx_emb=sx_emb,
                                      y=y, y_id=y_id.tolist(), y_emb=y_emb))
        return list_of_dicts

    def once_tree_sampling(self,
                           train_turns: List[List[Dict[str, Any]]],
                           test_turns: List[List[Dict[str, Any]]],
                           ) -> float:
        train_mms, test_mms = [], []
        total_logp_list: List[torch.Tensor] = []

        for turn in range(self.num_turns):
            train_samples = train_turns[turn]
            test_samples = test_turns[turn]

            logp_list, mm = self.xy_rv_mm(train_samples)
            total_logp_list += logp_list
            train_mms.append(mm)

            logp_list, mm = self.xy_rv_mm(test_samples)
            total_logp_list += logp_list
            test_mms.append(mm)

            self.logger.debug(f'turn {turn + 1} received the train/test mm:'
                              f'{train_mms[-1]}/{test_mms[-1]}')

        if self.num_turns < 2:
            reward = abs(fmean(train_mms) - fmean(test_mms))
        else:
            mean_train = fmean(train_mms)
            mean_test = fmean(test_mms)
            std_train = stdev(train_mms, mean_train)
            std_test = stdev(test_mms, mean_test)
            reward = abs(mean_train - mean_test) / std_train / std_test

        # the lower reward the better.
        loss = sum(total_logp_list) * reward
        loss.backward()
        return loss.item()

    def xy_rv_mm(self, samples: List[Dict[str, Any]]) -> Tuple[list, float]:
        logp_list, distances = [], []
        for sample in samples:
            tree_logp, dist = self.single_example(sample)
            logp_list.append(tree_logp)
            distances.append(dist)

        # this mean is just a quick evaluation for {x} and {y} due to its parallel
        xy_mm = sum(distances) / len(distances)
        return logp_list, xy_mm

    def single_example(self, sample: Dict[str, Any]) -> Tuple[torch.Tensor, float]:
        y_emb = sample['y_emb']
        prob_net = self._models[0]
        y_term_prob = prob_net(y_emb)  # (num_toks, 1)
        y_id_str = [str(x) for x in sample['y_id']]

        tree_logp, tree = TreeSampler().sample(y_id_str, y_term_prob.squeeze(-1))
        ty_emb = get_tree_emb(y_emb, tree).double()

        sx_emb = sample['sx_emb'].double()
        dist = ot.dist(sx_emb, ty_emb, metric='euclidean')
        null_t = torch.tensor([])
        return tree_logp, ot.emd2(null_t, null_t, dist).item()


class InducerEvalUpdater(InducerTrainingUpdater):
    @classmethod
    def from_bot(cls, bot: TrialBot) -> 'InducerEvalUpdater':
        args, p = bot.args, bot.hparams
        updater = super().from_bot(bot)
        iterators = [RandomIterator(len(d), p.batch_sz, False, False) for d in bot.datasets]
        updater._iterators = iterators
        updater._optims = []
        updater.ds_id = 0   # train by default
        return updater

    def set_inducing_ds(self, split: Literal['train', 'dev', 'test']):
        splits = ('train', 'dev', 'test')
        if split not in splits:
            raise ValueError(f'unsupported target dataset {split}, only {splits} available')
        self.ds_id = splits.index(split)

    def update_epoch(self):
        bot: TrialBot = self.bot # noqa
        bot.model.eval()
        ds = bot.datasets[self.ds_id]
        iter = self._iterators[self.ds_id]
        if iter.is_end_of_epoch:
            self.stop_epoch()

        samples: Tuple[List[str], List[str]] = self.get_samples(ds, iter)
        output: Dict[str, List[Any]] = self.induce_from_samples(samples)
        return output

    def induce_from_samples(self, samples: Tuple[List[str], List[str]]):
        with torch.no_grad():
            embedded_samples: List[Dict[str, Any]] = self.embed_data(*samples)
            mem = SeqCollector()
            keys = ('x', 'y', 'x_id', 'y_id', 'y_tree')
            for sample in embedded_samples:
                y_id_str = [str(x) for x in sample['y_id']]
                y_emb = sample['y_emb']
                prob_net = self._models[0]
                y_term_prob = prob_net(y_emb)  # (num_toks, 1)
                _, tree = TreeSampler(bounded_terms).sample(y_id_str, y_term_prob.squeeze(-1))
                mem(x=sample['x'], y=sample['y'],
                    x_id=sample['x_id'], y_id=sample['y_id'],
                    y_tree=tree)
            output = dict((k, mem[k]) for k in keys)
            return output


def get_tree_emb(term_embs: torch.Tensor, tree: Tree):
    """
    :param term_embs: (num_toks, dim)
    :param tree: a Tree instance, which have exactly num_toks leaves.
    :return: (num_tree_nodes, dim)
    """
    node_embs = []
    terminal_id = 0
    for n in PostOrderTraverse()(tree):
        n: Tree
        if n.is_terminal:
            emb = term_embs[terminal_id]  # (dim,)
            terminal_id += 1
        else:
            num_kids = len(n.children)
            emb = sum(node_embs[-num_kids:]) / num_kids  # (dim,)
        node_embs.append(emb)
        del emb

    tgt_emb = torch.stack(node_embs, dim=0)
    del node_embs
    return tgt_emb  # (num_nodes, dim)


def bernoulli_samples(span_prob) -> torch.Tensor:
    be_terms = span_prob.bernoulli()   # (span_length,)
    i = 0
    while be_terms.sum() < 1:
        del be_terms
        i += .1
        be_terms = (span_prob + i).clamp(max=.99).bernoulli()  # (span_length,)
    return be_terms


def bounded_terms(span_prob: torch.Tensor, num_max_terms: int = 3) -> torch.Tensor:
    """
    :param span_prob: (span_length,), values are [0, 1)
    :param num_max_terms:
    :return: (span_length,)
    """
    k = min(num_max_terms, len(span_prob) // 3)
    _, locations = torch.topk(span_prob, k, sorted=False)   # (k,)
    output = torch.zeros_like(span_prob)
    output[locations] = 1
    return output


class TreeSampler:
    def __init__(self, fn_get_terms: Callable = None, nt_label: str = 'nt'):
        """
        :param fn_get_terms: a callable function that reads span probabilities,
                    and produce an 1D binary vector of the span length,
                    indicating the token is a direct terminal (flag=1)
                    or contained in the subtree (flag=0)
        :param nt_label:
        """
        self.get_terms = fn_get_terms if fn_get_terms is not None else bernoulli_samples
        self.nt_label = nt_label

    def sample(self, terms, probs,
               span_begin: int = None,
               span_end: int = None,
               ) -> Tuple[torch.Tensor, Tree]:
        """
        :param probs:  (num_toks,)
        :param terms:   [str(id) of terms] len=num_toks
        :param span_begin: start of span for a tree
        :param span_end: end of span for a tree, not included
        :return: tuple of (tree_logp, tree)
        """
        span_begin = 0 if span_begin is None else span_begin
        span_end = len(terms) if span_end is None else span_end

        length = span_end - span_begin
        if length <= 3:
            return math.log(2), Tree(self.nt_label, children=[ # noqa
                Tree(terms[span_begin + i], is_terminal=True) for i in range(length)
            ])

        span_p = probs[span_begin:span_end]
        be_terms = self.get_terms(span_p)
        be_terms[0] = 1     # force no left-recursion

        level_logp = (be_terms * (span_p + 1).log() + (1 - be_terms) * (2 - span_p).log()).sum()
        terminal_locs = torch.argwhere(be_terms).squeeze(-1) + span_begin  # (num_immediate_children,)
        del be_terms

        sub_logp = []
        children = []
        for i, loc in enumerate(terminal_locs):     # iterate over only terminals
            # consider possible subtrees / nonterminals before the current terminal
            if i == 0 and loc > span_begin:
                # there's a subtree before the first terminal
                subtree_logp, subtree = self.sample(terms, probs, span_begin, loc)
                sub_logp.append(subtree_logp)
                children.append(subtree)

            elif loc > terminal_locs[i - 1] + 1:
                # a subtree between last and this terminals
                subtree_logp, subtree = self.sample(terms, probs, terminal_locs[i - 1] + 1, loc)
                sub_logp.append(subtree_logp)
                children.append(subtree)

            tree = Tree(terms[loc], is_terminal=True)
            children.append(tree)

            # a subtree after the last terminal, add it before ending the loop
            if i == len(terminal_locs) - 1 and loc < span_end - 1:
                subtree_logp, subtree = self.sample(terms, probs, loc + 1, span_end)
                sub_logp.append(subtree_logp)
                children.append(subtree)

        tree_logp = level_logp + sum(sub_logp, 0)
        tree = Tree(self.nt_label, children=children)
        return tree_logp, tree


@Registry.hparamset()
def base():
    p = HyperParamSet.common_settings(find_root('.ROOT'))
    p.OPTIM = 'adabelief'
    p.TRAINING_LIMIT = 100
    p.llm_path = osp.expanduser('~/.glm/chatglm-6b')
    p.max_tok_num = 1100
    p.hid_sz = 1024
    p.use_causal_lm_cls = False     # ChatGLM uses AutoModel instead of AutoModelForCausalLM

    p.batch_sz = 15
    p.num_turns = 5
    p.num_tree_sampling = 3
    p.ADAM_LR = 1e-3
    return p


if __name__ == '__main__':
    main()
