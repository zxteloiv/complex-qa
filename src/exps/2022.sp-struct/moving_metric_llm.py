import gc
import itertools
import json
import logging
import os
import random
import statistics
from functools import partial
import sys
from typing import List, Tuple, Callable, Union

import lark
import numpy as np
import transformers
from trialbot.data import Dataset
from trialbot.utils.root_finder import find_root
from trialbot.training.hparamset import HyperParamSet
from datetime import datetime as dt

sys.path.insert(0, find_root('.SRC'))

import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from trialbot.training import Registry, TrialBot
import os.path as osp
import ot

from shujuji import install_semantic_parsing_datasets, get_field_names_by_prefix
from utils.trialbot.setup_cli import setup as setup_cli, setup_null_argv
from utils.s2s_arch.hparam_modifiers import overwriting_mod_template, install_runtime_modifiers
from utils.lark.id_tree import build_from_lark_tree
from utils.tree import Tree, PreOrderTraverse, PostOrderTraverse


logger = logging.getLogger(__name__)


def batch_run():
    install_semantic_parsing_datasets()

    def _run_job(job):
        llm = mm = None
        for ds_key, p_key in itertools.product(job['datasets'], job['hparamsets']):
            args_dict = {
                "seed": 2021,
                "hparamset": p_key,
                "dataset": ds_key,
                "device": 0,
                "log-to-file": None,
            }
            args = setup_null_argv(**args_dict)
            llm, mm = main(args, llm, mm)

        del llm, mm

    job = {'datasets': ['cogs_gen_parsed', 'cogs_iid_parsed',
                        'smc128_parsed', 'smc64_parsed', 'smc128iid_parsed',
                        'ati_cg_handcrafted', 'ati_iid_handcrafted',
                        'adv_cg_handcrafted', 'adv_iid_handcrafted',
                        'geo_cg_handcrafted', 'geo_iid_handcrafted',
                        'sch_cg_handcrafted', 'sch_iid_handcrafted',
                        ],
           'hparamsets': ['falcon-tree', 'falcon-seq']}

    _run_job(job)


def main(args=None, llm=None, mm=None) -> Tuple['Embedder', 'MovingMetrics']:
    install_semantic_parsing_datasets()

    args = setup_cli(seed=2021, hparamset='mm-seq', device=0) if args is None else args

    install_runtime_modifiers(args.hparamset, partial(overwriting_mod_template, **dict(
      zip(('src_key', 'tgt_key'), get_field_names_by_prefix(args.dataset))
    )))
    train, dev, test = Registry.get_dataset(args.dataset)
    p = Registry.get_hparamset(args.hparamset)
    llm = Embedder(p.llm_path, args.device, p.max_tok_num, p.use_causal_lm_cls) if llm is None else llm
    mm = MovingMetrics(llm, p.num_retries, p.max_dist_size, p.ot_metric_string) if mm is None else mm

    key_x = p.src_key
    key_y = p.tgt_key + '_tree' if p.use_tgt_trees else p.tgt_key

    # potentially-large lists of either strings or trees
    train_x, train_y = get_distributions(train, key_x, key_y)
    # dev_x, dev_y = get_distributions(dev, key_x, key_y)
    test_x, test_y = get_distributions(test, key_x, key_y)
    logger.info(f'data size: x/x_: {len(train_x)}/{len(test_x)}')
    logger.info(f'data size: y/y_: {len(train_y)}/{len(test_y)}')

    with torch.inference_mode():
        logger.info(f'compute xy {timestr("%H:%M:%S")}')
        xy = mm.compute(train_x, train_y, parallel=True)
        logger.info(f'mm-output xy={xy}')

        logger.info(f'compute x_y_ {timestr("%H:%M:%S")}')
        x_y_ = mm.compute(test_x, test_y, parallel=True)
        logger.info(f'mm-output x_y_={x_y_}')

        logger.info(f'completed {timestr("%H:%M:%S")}')

    return llm, mm


def get_distributions(ds: Dataset, key_x: str, key_y: str):
    xs = []
    ys = []
    for example in ds:
        x = example.get(key_x)
        y = example.get(key_y)
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    return xs, ys


class Embedder:
    def __init__(self,
                 llm_path: str = '~/.glm/chatglm-6b',
                 device: int = -1,
                 max_tok_num: int = 1100,
                 use_causal_lm_cls: bool = False,
                 *,
                 llm=None,
                 tok=None,
                 ):

        if llm is not None and tok is not None:
            self.tok = tok
            self.model = llm
        else:
            llm_path = osp.expanduser(llm_path)
            self.tok = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
            # ChatGLM uses AutoModel, while Falcon and BaiChuan use AMForCausalLM
            auto_model_cls = AutoModelForCausalLM if use_causal_lm_cls else AutoModel
            self.model = auto_model_cls.from_pretrained(llm_path,
                                                        trust_remote_code=True,
                                                        torch_dtype=torch.float16,
                                                        device_map="auto")

        self.use_pad = self.tok.pad_token is not None
        self.model.eval()
        self.device = self.int_to_device(device)
        if device >= 0:
            self.model.cuda(device)
        self.max_tok_num = max_tok_num

    @staticmethod
    def int_to_device(device):
        if isinstance(device, torch.device):
            return device
        if device < 0:
            return torch.device("cpu")
        return torch.device(device)

    @torch.inference_mode()
    def embed(self, x_or_xs: Union[str, list]) -> Tuple[torch.DoubleTensor, torch.IntTensor]:
        x_inputs = self.tok(x_or_xs,
                            padding=self.use_pad,
                            return_token_type_ids=False,
                            return_tensors='pt').to(self.device)
        x_out = self.model(**x_inputs, output_hidden_states=True)
        x_emb = x_out.hidden_states[-1].double()
        if 'chatglm' in self.model.__class__.__name__.lower():
            x_emb = x_emb.transpose(0, 1)   # only the ChatGLM is not batch_first
        x_ids = x_inputs['input_ids']

        if isinstance(x_or_xs, list):
            return x_emb, x_ids
        else: # It'd accepted only a single string, and embedding without a batch form is expected
            return x_emb[0], x_ids[0]   # (len, dim), (len,) <- (batch, len, dim), (batch, len)

    @torch.inference_mode()
    def __call__(self, string_or_tree: Union[str, lark.Tree], on_tgt_side: bool = False) -> torch.Tensor:
        if isinstance(string_or_tree, str):
            if len(string_or_tree) == 0:
                raise ValueError('Empty string cannot be fed and embedded.')
            x_emb, x_id = self.embed(string_or_tree)
            if on_tgt_side:
                return self.compute_right_branching_emb(x_emb, x_id)
            else:
                return self.compute_flat_seq_emb(x_emb, x_id)

        elif isinstance(string_or_tree, lark.Tree):
            tree = build_from_lark_tree(string_or_tree)
            terms: str = self.restore_text_from_trees(tree)

            terms_emb, terms_id = self.embed(terms)
            return self.compute_tree_style_emb(terms_id, terms_emb, tree)

        else:
            raise TypeError('Invalid list. Only a single string or a Lark tree is supported.')

    @torch.inference_mode()
    def compute_flat_seq_emb(self, emb, x_ids):
        """
        :param emb: (padded_tok_num, dim)
        :param x_ids: (padded_tok_num,)
        :return: (tok_num + 1, dim)
        """
        # mask = (x_ids > 20004).unsqueeze(-1)   # printable tokens starts from 20005
        # hid_sz = emb.size(-1)
        # selected_embs = torch.masked_select(emb, mask).reshape(-1, hid_sz)[:self.max_tok_num]
        selected_embs = emb     # suppose no pad is included
        sent_emb = selected_embs.mean(0).unsqueeze(0)   # (1, dim)
        src_emb = torch.cat([selected_embs, sent_emb], dim=0)   # (sel_toks + 1, dim)
        return src_emb

    @torch.inference_mode()
    def compute_right_branching_emb(self, emb, x_ids):
        """
        Assuming a tree NT node is represented as the mean of the children, with the right-branching tree.

        :param emb: (padded_tok_num, dim)
        :param x_ids: (padded_tok_num,)
        :return: (tok_num + (tok_num - 1), dim), the order doesn't matter.
        """
        # mask = (x_ids > 20004).unsqueeze(-1)   # printable tokens starts from 20005
        # hid_sz = emb.size(-1)
        # selected_emb = torch.masked_select(emb, mask).reshape(-1, hid_sz)[:self.max_tok_num]   # (tok_num, dim)
        selected_emb = emb     # suppose no pad is included

        nt_embs = []
        last_emb = selected_emb[-1]
        for emb in reversed(selected_emb[:-1]):
            node_emb = (emb + last_emb) / 2
            nt_embs.append(node_emb)
            last_emb = node_emb
            del node_emb

        return torch.cat([torch.stack(nt_embs, dim=0), selected_emb], dim=0)

    @staticmethod
    def compute_offsets(tree):
        running_prefix = 0
        for node in PostOrderTraverse()(tree):
            if node.is_terminal:
                node.payload = (running_prefix, running_prefix + len(node.label))
                running_prefix += 1 + len(node.label)  # the space sep
            else:
                # use the left-most to right-most
                node.payload = (node.children[0].payload[0], node.children[-1].payload[-1])
        return tree

    @torch.inference_mode()
    def compute_tree_style_emb(self, term_ids, term_embs, tree):
        """
        Assuming a tree NT node is represented as the mean of the children.
        :param term_ids: (tok_num,)
        :param term_embs: (tok_num, dim)
        :param tree: a Tree instance
        :return: (num_nodes, dim)
        """
        node_embs = []

        tree = self.compute_offsets(tree)   # update the payload, indicating offsets to the original text for nodes

        if isinstance(self.tok, transformers.PreTrainedTokenizerFast):
            len_contrib = self.tok(self.restore_text_from_trees(tree), return_offsets_mapping=True)['offset_mapping']
            len_contrib = [end for _, end in len_contrib]
        else:
            len_contrib: list = [len(self.tok.decode(term_ids[:i + 1])) for i in range(len(term_ids))
                                 if i < self.max_tok_num]

        def _find_tok_id(n) -> int:
            if n >= len_contrib[-1]:
                return -1

            i_offset = 0
            for i, contrib in enumerate(len_contrib):
                if i_offset <= n < contrib:
                    return i
                i_offset = contrib

            # raise ValueError(f'invalid n={n} exceeds the input length {len_contrib[-1]}')
            return -1

        for node in PostOrderTraverse()(tree):
            node: Tree
            if node.is_terminal:
                start_offset, end_offset = node.payload     # [start, end)
                start_id = _find_tok_id(start_offset)       # start_id,
                end_id = _find_tok_id(end_offset - 1)       # and end_id are included
                emb = term_embs[start_id:end_id + 1].mean(dim=0)    # (dim,)

            else:
                num_children = len(node.children)
                emb = sum(node_embs[-num_children:]) / num_children     # (dim,)

            node_embs.append(emb)
            del emb

        tgt_emb = torch.stack(node_embs, dim=0)
        del node_embs
        return tgt_emb  # (num_nodes, dim)

    @staticmethod
    def restore_text_from_trees(t_or_ts):
        def _restore(t):
            return ' '.join(node.label for node in PreOrderTraverse()(t) if node.is_terminal)

        if isinstance(t_or_ts, Tree):
            return _restore(t_or_ts)
        else:
            return [_restore(t) for t in t_or_ts]


class MovingMetrics:
    def __init__(self,
                 embed: Embedder,
                 num_retries: int = 0,
                 max_elem_num: int = 0,
                 ot_metric_string: str = 'euclidean',
                 ):
        self.num_retries: int = num_retries
        self.max_elem_num: int = max_elem_num
        self.metric = ot_metric_string
        self.embed = embed

        self._row_first: bool = False   # set to False because the Y's are usually longer
        self._last_ts = None
        self._last_emb = None

    def clear_cache(self):
        self._last_ts = None
        self._last_emb = None
        if self.embed.device.type != 'cpu':
            torch.cuda.empty_cache()

    def cost_on_distributions(self, xs: list, xt: list):
        n, m = len(xs), len(xt)
        logger.info(f'distribution costs estimating: {n}/{m}. {timestr()}')
        costs = torch.zeros(n, m, device=self.embed.device, dtype=torch.double)

        if self._row_first:
            prod = itertools.product(range(n), range(m))
        else:
            prod = itertools.product(range(m), range(n))

        # row first will keep rows the same in consecutive calls, the opposite otherwise.
        for i, j in prod:
            if not self._row_first:
                i, j = j, i

            cost_ij = self.direct_metric(xs[i], xt[j])
            costs[i, j] = cost_ij
            del cost_ij
        return costs

    def _retrieve_and_save(self, ts, for_tgt: bool = False):
        if not hasattr(self, '_last_ts'):
            self._last_ts = None
            self._last_emb = None

        if self._last_ts == ts:
            emb = self._last_emb
        else:
            emb = self.embed(ts, for_tgt)
            del self._last_emb, self._last_ts
            self._last_ts = ts
            self._last_emb = emb
        return emb

    def cost_on_tensors(self, xs, xt):
        if self._row_first:
            xs_emb: torch.Tensor = self._retrieve_and_save(xs)
            xt_emb: torch.Tensor = self.embed(xt, on_tgt_side=True)
        else:
            xs_emb: torch.Tensor = self.embed(xs)
            xt_emb: torch.Tensor = self._retrieve_and_save(xt, for_tgt=True)

        n, m = xs_emb.size(0), xt_emb.size(0)
        logger.info(f'tensor costs estimating: {n}/{m}. {timestr()}')
        costs = ot.dist(xs_emb, xt_emb, metric=self.metric)  # perhaps a tensor on cuda
        return costs

    def direct_metric(self, xs, xt, parallel: bool = False) -> torch.Tensor:
        """
        Compute the metric. Exceptions like scales and lengths are assured absent.
        Both distribution must be in the same metric space.
        :param xs: list of (str or lark.Tree)
        :param xt: list of (str or lark.Tree)
        :param parallel: bool, if True, the xs and xt are parallel samples,
                        computing the average metric for each pair of them is just fine.
                        This will save a lot computations.
        :return:
        """
        if isinstance(xs, list) and parallel:
            logger.info(f'parallel metric: sizes={len(xs)}/{len(xt)}, parallel={parallel}: {timestr()}')
            pairwise_metrics = []
            for a, b in zip(xs, xt):
                pairwise_metrics.append(self.direct_metric(a, b, parallel=False))
            return sum(pairwise_metrics) / len(pairwise_metrics)

        if isinstance(xs, list):
            costs = self.cost_on_distributions(xs, xt)
        else:
            costs = self.cost_on_tensors(xs, xt)

        self.clear_cache()
        return ot.emd2(torch.tensor([]), torch.tensor([]), costs)   # a torch scalar

    def sample(self, *args):   # sample from all sources based on the first
        if len(args[0]) <= self.max_elem_num:
            if len(args) == 1: return args[0]
            else: return args
        else:
            indices = random.sample(range(len(args[0])), k=self.max_elem_num)
            if len(args) == 1: return [args[0][i] for i in indices]
            else: return tuple([a[i] for i in indices] for a in args)

    def set_num_retries(self, val: int):
        self.num_retries = val

    def set_max_elem_num(self, val: int):
        self.max_elem_num = val

    def compute(self, xs: list, xt: list, parallel: bool = False) -> Tuple[float, float]:
        if len(xs) <= 0 or len(xt) <= 0:
            raise ValueError('Input samples must not be empty.')

        if parallel and len(xs) != len(xt):
            raise ValueError('Input samples are of different size and thus not parallel.')

        if self.max_elem_num <= 0:  # the option is valid only when it's positive
            return self.direct_metric(xs, xt, parallel=parallel).item(), 1.

        # when both distributions are small, return the exact direct metric.
        if len(xs) <= self.max_elem_num and len(xt) <= self.max_elem_num:
            return self.direct_metric(xs, xt, parallel=parallel).item(), 1.

        metrics = []
        for turn in range(self.num_retries):
            if parallel:  # in parallel both xs and xt are bounded in size.
                xss, xtt = self.sample(xs, xt)

            else:   # decoupled sampling
                xss = self.sample(xs)
                xtt = self.sample(xt)

            metrics.append(self.direct_metric(xss, xtt, parallel=parallel).item())
            logger.info(f'turn {turn+1}/{self.num_retries} metric: {metrics[-1]}')

        return statistics.fmean(metrics), statistics.stdev(metrics)


def timestr(fmt='%H:%M:%S'):
    return dt.now().strftime(fmt)


@Registry.hparamset('mm-seq')
def base():
    p = HyperParamSet()

    # embedder params
    p.llm_path = osp.expanduser('~/.glm/chatglm-6b')
    p.max_tok_num = 1100    # >num of 99% examples on ATIS

    # moving-metric params
    p.num_retries = 10
    p.max_dist_size = 50
    p.ot_metric_string = 'euclidean'

    p.use_tgt_trees = False

    p.use_causal_lm_cls = False
    return p


@Registry.hparamset('mm-tree')
def dump():
    p = base()
    p.use_tgt_trees = True
    return p


@Registry.hparamset('falcon-seq')
def falcon7b():
    p = base()
    # p.llm_path = 'tiiuae/falcon-7b'
    p.llm_path = osp.expanduser('~/.cache/manual_llm_cache/falcon-7b')
    p.use_causal_lm_cls = True
    return p


@Registry.hparamset('falcon-tree')
def falcon7b_tree():
    p = falcon7b()
    p.use_tgt_trees = True
    return p


@Registry.hparamset('baichuan-seq')
def baichuan7b():
    p = base()
    # p.llm_path = 'baichuan-inc/baichuan-7B'
    p.llm_path = osp.expanduser('~/.cache/manual_llm_cache/baichuan-7b')
    p.use_causal_lm_cls = True
    return p


@Registry.hparamset('baichuan-tree')
def baichuan7b_tree():
    p = baichuan7b()
    p.use_tgt_trees = True
    return p


if __name__ == '__main__':
    batch_run()
    # main()
