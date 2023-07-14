# A tree prior aims to incorporate our general (but not exact) knowledge about tree grammars.
#
import logging
import math

import trialbot.utils.prepend_pythonpath    # noqa
from operator import itemgetter
from typing import Any, Callable, cast

import ot
import torch
import trialbot.data
from trialbot.data import RandomIterator
from trialbot.utils.root_finder import find_root
from statistics import fmean, stdev
import os.path as osp
from trialbot.training import TrialBot, Updater, Events
from trialbot.training import Registry
from trialbot.training.hparamset import HyperParamSet
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from allennlp.common.util import int_to_device
from allennlp.nn.util import masked_mean

from shujuji import install_semantic_parsing_datasets
from utils.trialbot.dummy_translator import install_dummy_translator
from utils.trialbot.setup_bot import setup_bot
from utils.trialbot.setup_cli import setup as setup_cli
from models.base_s2s.rnn_stacker import RNNCellStacker
from models.onlstm.onlstm import ONLSTMCell
from models.modules.variational_dropout import VariationalDropout as VD
from utils.select_optim import select_optim
from shujuji import get_field_names_by_prefix
from utils.tree import Tree, PreOrderTraverse, PostOrderTraverse

T = torch.Tensor


def main():
    # add configurations to the Registry
    install_semantic_parsing_datasets()
    install_dummy_translator()
    install_hparams()

    args = setup_cli(seed=2021, translator='dummy', hparamset='base-prior', device=0)
    bot = TrialBot(args=args, trial_name='tree_prior', get_model_func=TreeInducer.new)
    bot = setup_bot(bot, True, False, False, False, True, True)
    bot.updater = PolicyUpdater.from_bot(bot)

    from trialbot.training.extensions import loss_reporter
    bot._engine.remove_event_handler(loss_reporter, Events.ITERATION_COMPLETED)
    bot.add_event_handler(Events.ITERATION_COMPLETED, loss_reporter, 100, interval=1)

    @bot.attach_extension(Events.EPOCH_COMPLETED)
    def run_eval(bot: TrialBot):
        if bot.state.iteration % 20 == 0:
            updater = cast(PolicyUpdater, bot.updater)
            metric = updater.eval()
            bot.logger.info(f'Eval on dev got the metric {metric}.')

    bot.run()


def install_hparams():
    @Registry.hparamset('base-prior')
    def base():
        p = HyperParamSet.common_settings(find_root())
        p.TRAINING_LIMIT = 1
        p.GRAD_CLIPPING = 2
        p.batch_sz = 16
        p.llm_path = osp.expanduser('~/.glm/chatglm-6b')
        p.llm_hid_sz = 4096
        p.hid_sz = 1024
        p.chunk_sz = 32
        p.dropout = .01
        p.num_tree_sampling = 5
        return p


class TreeInducer(RNNCellStacker):
    """The tree inducer subtyping an RNN stack of ON-LSTM cells
    extending the method to infer trees based on the ON-LSTM internals."""
    @classmethod
    def new(cls, p: trialbot.training.hparamset.HyperParamSet, vocab: trialbot.data.NSVocabulary) -> 'TreeInducer':
        return cls([
            ONLSTMCell(p.llm_hid_sz, p.hid_sz, p.chunk_sz, VD(p.dropout, on_the_fly=False)),
        ])

    def get_tree_dist(self, emb: T, mask: T | None = None, ) -> tuple[T, T]:
        """
        :param emb: (seq_len, emb)
        :param mask: (seq_len,) or None
        :return tuple of: tree log-prob, and terminal embeddings
                  logp: (#nodes, #chunk)
            embeddings: (#nodes, emb), the padded nodes are NOT included.
        """
        hx: None | list[tuple[T, T, T, T, T]] = None
        df_list: list[T] = []
        term_embs = []
        for step_in, step_mask in zip(emb, mask):
            if step_mask > 0:   # assuming the non-padded tokens are consecutive
                hx, _ = self(step_in[None, :], hx)
                # df_logits: (batch_sz=1, n_chunk)
                df_logits = hx[0][-1]  # we have only one cell in the rnn stack in fact.
                df_list.append(df_logits)
                term_embs.append(step_in)   # (emb,)
                logging.debug(f'step: {step_in.size()}, df_logits: {df_logits.size()}')

        dfs = torch.cat(df_list, dim=0)     # (#non-padded-nodes, #chunk)
        term_emb = torch.stack(term_embs)   # (#non-padded-nodes, emb)
        del term_embs
        logp_df = torch.log_softmax(dfs, dim=-1)
        return logp_df, term_emb

    def infer_trees(self, logp_df: T, n_sampled_trees: int = 1) -> tuple[list[Tree], T]:
        # logp_df: (#nodes, #chunk)
        dev = logp_df.device
        # we sample several trees for a single terminal
        noise = torch.distributions.Gumbel(
            torch.tensor(0., device=dev),
            torch.tensor(1., device=dev)
        ).sample((n_sampled_trees,) + logp_df.size())   # (#trees, #non-padded, #chunk)

        # each df indicates the level of a terminal, and higher levels mean closer to the root
        sampled_df = (logp_df[None, :, :] + noise).argmax(dim=-1)   # (#trees, #non-padded)
        n_trees, n_nodes = sampled_df.size()

        def tr(n: int): return torch.arange(n, device=dev)

        # logp: (#trees, #nodes),
        sampled_logp: T = logp_df[tr(n_nodes)[None, :], sampled_df]

        # tree_logp: (#trees,)
        # trees: list[Tree]
        sampled_tree_logp = sampled_logp.sum(dim=-1)
        sampled_trees = [self.greedy_tree_from_df(df.tolist()) for df in sampled_df]

        return sampled_trees, sampled_tree_logp

    def greedy_tree_from_df(self, df: list[int]) -> Tree:
        # df: (#non-padded,)
        # start and end are global indices to the whole sentence
        # conventional span range, [start, end)

        def _rec_tree(start: int, end: int) -> Tree:
            assert start < end
            if start == end - 1:
                return Tree(str(start), is_terminal=True)

            max_loc: int = max(range(end - start), key=lambda x: df[x]) + start

            if max_loc == start:
                return Tree('nt', children=[
                    Tree(str(max_loc), is_terminal=True),
                    _rec_tree(max_loc + 1, end),
                ])
            elif max_loc + 1 == end:
                return Tree('nt', children=[
                    _rec_tree(start, max_loc),
                    Tree(str(max_loc), is_terminal=True),
                ])
            else:
                return Tree('nt', children=[
                    _rec_tree(start, max_loc),
                    Tree(str(max_loc), is_terminal=True),
                    _rec_tree(max_loc + 1, end),
                ])

        return _rec_tree(0, len(df))

    @staticmethod
    def get_tree_emb(term_embs: T, tree: Tree):
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


class PriorAgent:
    """An entity that knows how to use the tree inducer model.
    But it must not be interwoven with training procedure."""
    def __init__(self, inducer: TreeInducer, llm_path: str, device: int, n_tree_samples: int = 1):
        llm_path = osp.expanduser(llm_path)
        self.tok = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        # ChatGLM uses AutoModel, while Falcon and BaiChuan use AMForCausalLM
        self.llm = AutoModel.from_pretrained(llm_path,
                                             trust_remote_code=True,
                                             # torch_dtype=torch.float16,
                                             ).half().eval()
        self.device = int_to_device(device)
        if device >= 0:
            self.llm.cuda(self.device)
        self.inducer = inducer
        self.logger = logging.getLogger(self.__class__.__name__)
        self.n_trees = n_tree_samples

    def embed(self, x_or_xs: str | list[str]) -> tuple[torch.DoubleTensor, torch.IntTensor]:
        with torch.inference_mode():
            x_inputs = self.tok(x_or_xs,
                                padding=True,
                                return_token_type_ids=False,
                                return_tensors='pt').to(self.device)
            x_out = self.llm(**x_inputs, output_hidden_states=True)
            x_emb = x_out.hidden_states[-1]
            if 'chatglm' in self.llm.__class__.__name__.lower():
                x_emb = x_emb.transpose(0, 1)   # only the ChatGLM is not batch_first
            x_ids = x_inputs['input_ids']

            if not isinstance(x_or_xs, list):
                # (len, dim), (len,) <- (batch, len, dim), (batch, len)
                x_emb = x_emb[0]
                x_ids = x_ids[0]

        return x_emb.float().clone().detach_(), x_ids.clone().detach_()

    def embed_data(self, x_data: list[str], y_data: list[str]) -> list[dict[str, Any]]:
        list_of_dicts: list[dict[str, Any]] = []

        for x, y in zip(x_data, y_data):
            x_emb, x_id = self.embed(x)     # (num_toks, emb)
            sx_emb = self.flat_tree_emb(x_emb, self.get_mask(x_id))
            y_emb, y_id = self.embed(y)     # (num_toks, emb)

            list_of_dicts.append(dict(x=x, x_id=x_id, x_emb=x_emb, sx_emb=sx_emb,
                                      y=y, y_id=y_id, y_emb=y_emb))
        return list_of_dicts

    def get_mask(self, x_id) -> torch.BoolTensor:
        if 'chatglm' in self.llm.__class__.__name__.lower():
            return x_id > 20004     # for ChatGLM only, the printable tokens start from 20005

        raise NotImplementedError

    def flat_tree_emb(self, x_emb: T, x_mask: torch.BoolTensor | None = None) -> T:
        """
        :param x_emb: (seq, dim)
        :param x_mask: (seq,), mask of 0 and 1 for the corresponding embeddings
        :return: (#nodes, dim), padded nodes are excluded,
                #nodes is #non_padded_nodes + 1 in this flat tree setting.
        """
        with torch.inference_mode():
            if x_mask is None:
                root = x_emb.mean(0, keepdim=True)  # (1, dim)
                flat_emb = torch.cat([root, x_emb], dim=0)  # (seq+1, dim)
            else:
                x_mask = x_mask[:, None]
                emb_sz = x_emb.size()[-1]
                root = masked_mean(x_emb, x_mask, dim=0, keepdim=True)          # (1, dim)
                nodes = torch.masked_select(x_emb, x_mask).view(-1, emb_sz)     # (#non-padded, dim)
                flat_emb = torch.cat([root, nodes], dim=0)

        return flat_emb.clone().detach_()

    def rl_loss(self, train_samples: list[dict[str, Any]], test_samples: list[dict[str, Any]]) -> T:
        """Compute the moving metric of X and Y random variables.
        Because Xs and Ys are parallel corpus, they have to be transported one on one.
        Therefore, the moving metric between X and Y rvs is actually
        the mean of cost to transport every single (X, Y) pair.
        """
        def _rl(ts: tuple[T, T]) -> T:  # reward x log-prob, both of (#tree,)
            return (ts[0] * ts[1]).mean()   # all tree averaged

        train_losses = [_rl(self.single_example(sample)) for sample in train_samples]
        test_losses = [_rl(self.single_example(sample)) for sample in test_samples]

        # this mean is just a quick evaluation for {x} and {y} because they're parallel
        mm_train = sum(train_losses) / len(train_losses)
        mm_test = sum(test_losses) / len(test_losses)

        # the lower discrepancy the better structure
        rl_loss = (mm_train - mm_test).abs()
        return rl_loss

    def single_example(self, sample: dict[str, Any], n_trees: int | None = None) -> tuple[T, T]:
        y_emb = sample['y_emb']
        logp_df, term_embs = self.inducer.get_tree_dist(y_emb, self.get_mask(sample['y_id']))
        trees, trees_logp = self.inducer.infer_trees(logp_df, self.n_trees if n_trees is None else n_trees)

        sx_emb = sample['sx_emb']
        dists: list[T] = []
        for tree in trees:
            ty_emb = self.inducer.get_tree_emb(term_embs, tree)
            dist = ot.dist(sx_emb, ty_emb, metric='euclidean')
            null_t = torch.tensor([])
            moving_dist = ot.emd2(null_t, null_t, dist).detach()
            dists.append(moving_dist)

        trees_dist = torch.stack(dists)
        return trees_dist, trees_logp   # both of (#trees,)


class PolicyUpdater(Updater):
    @classmethod
    def from_bot(cls, bot: TrialBot) -> 'Updater': return cls(bot)

    def __init__(self, bot: TrialBot):
        args, p = bot.args, bot.hparams
        iterators = list(map(lambda d: RandomIterator(len(d), p.batch_sz), bot.datasets))
        optim = select_optim(p, bot.model.parameters())
        super().__init__(bot.models, iterators, optim)

        self.bot = bot   # save the backward reference for convenience
        self.key_pair = get_field_names_by_prefix(args.dataset)
        self.grad_clip = p.GRAD_CLIPPING
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agent = PriorAgent(bot.model, p.llm_path, args.device, p.num_tree_sampling)

    def update_epoch(self) -> dict[str, Any]:
        datasets = self.bot.datasets    # noqa
        iterators = self._iterators

        # sample and embed data, which is fixed during each time of tree sampling
        train_samples = self.get_samples(datasets[0], iterators[0])
        test_samples = self.get_samples(datasets[1], iterators[1])  # use dev for prior training
        if iterators[0].is_end_of_epoch:
            self.stop_epoch()

        embedded_train = self.agent.embed_data(*train_samples)
        embedded_test = self.agent.embed_data(*test_samples)

        loss = self.agent.rl_loss(embedded_train, embedded_test)
        optim = self._optims[0]
        optim.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_value_(self.bot.model.parameters(), self.grad_clip)
        optim.step()
        return {'loss': loss}

    def get_samples(self, dataset, iterator) -> tuple[list[Any], list[Any]]:
        indices = next(iterator)
        translator = self.bot.translator
        samples: dict[str, list[Any]] = translator.batch_tensor([translator.to_tensor(dataset[i]) for i in indices])
        return itemgetter(*self.key_pair)(samples)

    def eval(self) -> float:
        """Provide the evaluation metric (the defined moving metric) instead of RL loss."""
        p = self.bot.hparams
        ds = self.bot.datasets
        ris = [RandomIterator(len(d), p.batch_sz) for d in ds]

        T_X = T_Y = list[str]
        T_DATA = tuple[T_X, T_Y]
        T_TURN = tuple[T_DATA, T_DATA]  # train and test data
        multiturn_data: list[T_TURN] = []
        n_turns = math.ceil(50 / p.batch_sz)
        for turn in range(n_turns):
            train = self.get_samples(ds[0], ris[0])
            test = self.get_samples(ds[1], ris[1])  # in fact using the dev
            multiturn_data.append((train, test))

        mm_train, mm_test = [], []
        with torch.inference_mode():
            for train, test in multiturn_data:
                mm_train.append(fmean([self.agent.single_example(s)[0].mean().item()
                                       for s in self.agent.embed_data(*train)]))    # metric of parallel training data
                mm_test.append(fmean([self.agent.single_example(s)[0].mean().item()
                                      for s in self.agent.embed_data(*test)]))      # metric of parallel testing data

        metric = abs(fmean(mm_train) - fmean(mm_test)) / stdev(mm_train) / stdev(mm_test)
        return metric


if __name__ == "__main__":
    main()
