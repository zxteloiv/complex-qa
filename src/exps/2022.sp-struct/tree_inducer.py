import os
from functools import partial
from operator import itemgetter
from statistics import fmean, stdev
from typing import Tuple, Union, List

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
from utils.select_optim import select_optim
from utils.s2s_arch.hparam_modifiers import install_runtime_modifiers, param_overwriting_modifier
from utils.tree import Tree, PreOrderTraverse, PostOrderTraverse

from moving_metric_llm import MovingMetrics, Embedder


def main():
    install_semantic_parsing_datasets()
    install_dummy_translator(filter_none=True)
    if os.getenv('CUDA_VISIBLE_DEVICES') is None:
        raise OSError("Using accelerate package requires manually setting the ENV variable.")

    args = setup_cli(device=0, seed=2021, translator='dummy', hparamset='base')

    bot = TrialBot(args, 'tree_inducer', get_model)

    @bot.attach_extension(Events.STARTED)
    def print_llm(bot):
        bot.logger.info("LLM Spec:\n" + str(bot.embedder.model))
        bot.logger.info("Tokenizer Spec:\n" + str(bot.embedder.tok))

    # init additional components
    ds_field_names = get_field_names_by_prefix(args.dataset)
    bot.translator.gather_keys = ds_field_names
    bot.embedder = get_moving_metric(bot)
    bot.updater = MyUpdater.from_bot(bot)

    bot.run()


def get_model(p, vocab):
    # get log prob(t): (num_toks, 1) <- (num_toks, llm_hid_sz)
    return torch.nn.Sequential(
        torch.nn.Linear(4096, p.hid_sz),
        torch.nn.Mish(),
        torch.nn.Linear(p.hid_sz, 1),
        torch.nn.LogSigmoid()
    )


def get_moving_metric(bot):
    args, p = bot.args, bot.hparams
    embedder = Embedder(p.llm_path, device=args.device, max_tok_num=p.max_tok_num,
                        use_causal_lm_cls=p.use_causal_lm_cls,)
    return embedder


class MyUpdater(Updater):
    @classmethod
    def from_bot(cls, bot: TrialBot) -> 'Updater':
        args, p = bot.args, bot.hparams
        iterators = list(map(lambda d: RandomIterator(len(d), p.batch_sz), bot.datasets))
        optim = select_optim(p, bot.model.parameters())
        updater = cls(bot.models, iterators, [optim], device=args.device)
        updater.bot = bot   # save the backward reference for convenience
        updater.num_turns = p.num_turns
        updater.num_k = p.num_tree_sampling
        return updater

    def update_epoch(self):
        datasets = self.bot.datasets    # noqa

        # sample data, fixed during each tree sampling
        train_samples, test_samples = [], []
        for turn in range(self.num_turns):
            train_samples.append(self.get_samples(datasets[0], self._iterators[0]))
            test_samples.append(self.get_samples(datasets[-1], self._iterators[-1]))

        if self._iterators[-1].is_end_of_epoch:
            self.stop_epoch()

        optim = self._optims[0]
        optim.zero_grad()
        losses = []
        for k in range(self.num_k):
            loss_k = self.once_tree_sampling(train_samples, test_samples)
            losses.append(loss_k)
        optim.step()
        return {'loss': sum(losses)}

    def get_samples(self, dataset, iterator):
        indices = next(iterator)
        translator = self.bot.translator
        samples = translator.batch_tensor([translator.to_tensor(dataset[i]) for i in indices])
        key_pairs = get_field_names_by_prefix(self.bot.args.dataset)
        return itemgetter(*key_pairs)(samples)

    def once_tree_sampling(self, train_samples, test_samples) -> float:
        logp_list, reward = self.single_tree(train_samples, test_samples)
        loss = - sum(logp_list) * reward
        loss.backward()
        return loss.item()

    def single_tree(self, train_data, test_data) -> Tuple[List[torch.Tensor], float]:
        if self.num_turns == 1:
            train_logp_list, train_mm = self.single_turn(*train_data[0])
            test_logp_list, test_mm = self.single_turn(*test_data[0])
            reward = abs(train_mm - test_mm)
            return train_logp_list + test_logp_list, reward

        logp_list = []
        train_mm_list, test_mm_list = [], []
        for turn in range(self.num_turns):
            turn_logp_list, train_mm = self.single_turn(*train_data[turn])
            logp_list.extend(turn_logp_list)
            train_mm_list.append(train_mm)

            turn_logp_list, test_mm = self.single_turn(*test_data[turn])
            logp_list.extend(turn_logp_list)
            test_mm_list.append(test_mm)

        mean_train = fmean(train_mm_list)
        mean_test = fmean(test_mm_list)
        std_train = stdev(train_mm_list)
        std_test = stdev(test_mm_list)

        reward = abs(mean_train - mean_test) / std_train / std_test
        return logp_list, reward

    def single_turn(self, xs, ys) -> Tuple[list, float]:
        logp_list, distances = [], []
        for x, y in zip(xs, ys):
            tree_logp, dist = self.single_example(x, y)
            logp_list.append(tree_logp)
            distances.append(dist)

        # this mean is just a quick evaluation for {x} and {y} due to its parallel
        xy_mm = sum(distances) / len(distances)
        return logp_list, xy_mm

    def single_example(self, x, y) -> Tuple[torch.Tensor, float]:
        embedder: Embedder = self.bot.embedder    # noqa

        x_emb, x_id = embedder.embed(x)  # (num_toks, emb)
        x_emb = x_emb.float().clone()
        sx_emb = embedder.compute_flat_seq_emb(x_emb, x_id)

        y_emb, y_id = embedder.embed(y)  # (num_toks, emb)
        y_emb = y_emb.float().clone()
        logit_net = self._models[0]
        y_contrib_logp = logit_net(y_emb)    # (num_toks, 1)
        tree_logp, tree = sample_a_tree([str(x) for x in y_id.tolist()], y_contrib_logp.squeeze(-1))
        ty_emb = get_tree_emb(y_emb, tree)

        dist = ot.dist(sx_emb.double(), ty_emb.double(), metric='euclidean')
        null_t = torch.tensor([])
        return tree_logp, ot.emd2(null_t, null_t, dist).item()


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


def sample_a_tree(terms, logp,
                  span_begin: int = None,
                  span_end: int = None,) -> Tuple[torch.Tensor, Tree]:
    """
    :param logp:  (num_toks,)
    :param terms:   [str(id) of terms] len=num_toks
    :param span_begin: start of span for a tree
    :param span_end: end of span for a tree, not included
    :return: tuple of (tree_logp, tree)
    """
    nt_label = 'NT'
    span_begin = 0 if span_begin is None else span_begin
    span_end = len(terms) if span_end is None else span_end

    length = span_end - span_begin
    if length <= 3:
        return 0, Tree(nt_label, children=[ # noqa
            Tree(terms[span_begin + i], is_terminal=True) for i in range(length)
        ])

    density = logp[span_begin:span_end].exp()
    be_terms = density.bernoulli()   # (span_length,)
    i = 0
    while be_terms.sum() < 1:
        del be_terms
        i += .1
        be_terms = (density + i).clamp(max=.99).bernoulli()  # (span_length,)

    level_logp = (be_terms * logp[span_begin:span_end]).sum()
    terminal_locs = torch.argwhere(be_terms).squeeze(-1) + span_begin  # (num_immediate_children,)
    del be_terms

    sub_logp = []
    children = []
    for i, loc in enumerate(terminal_locs):     # iterate over only terminals
        # consider possible subtrees / nonterminals before the current terminal
        if i == 0 and loc > span_begin:
            # there's a subtree before the first terminal
            subtree_logp, subtree = sample_a_tree(terms, logp, span_begin, loc)
            sub_logp.append(subtree_logp)
            children.append(subtree)

        elif loc > terminal_locs[i - 1] + 1:
            # a subtree between last and this terminals
            subtree_logp, subtree = sample_a_tree(terms, logp, terminal_locs[i - 1] + 1, loc)
            sub_logp.append(subtree_logp)
            children.append(subtree)

        tree = Tree(terms[loc], is_terminal=True)
        children.append(tree)

        # a subtree after the last terminal, add it before ending the loop
        if i == len(terminal_locs) - 1 and loc < span_end - 1:
            subtree_logp, subtree = sample_a_tree(terms, logp, loc + 1, span_end)
            sub_logp.append(subtree_logp)
            children.append(subtree)

    tree_logp = level_logp + sum(sub_logp, 0)
    tree = Tree(nt_label, children=children)
    return tree_logp, tree


@Registry.hparamset()
def base():
    p = HyperParamSet.common_settings(find_root('.ROOT'))
    p.OPTIM = 'adabelief'
    p.llm_path = osp.expanduser('~/.glm/chatglm-6b')
    p.max_tok_num = 1100
    p.hid_sz = 1024
    p.use_causal_lm_cls = False # ChatGLM uses AutoModel instead of AutoModelForCausalLM

    p.batch_sz = 50
    p.num_turns = 10
    p.num_tree_sampling = 5
    p.ADAM_LR = 1e-3
    return p


if __name__ == '__main__':
    main()
