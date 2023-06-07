from functools import partial
import sys
from trialbot.utils.root_finder import find_root
import logging

sys.path.insert(0, find_root('.SRC'))

import torch
from transformers import AutoModel, AutoTokenizer
from trialbot.training import Registry, TrialBot, Events
import os.path as osp
import ot

from shujuji import install_semantic_parsing_datasets, get_field_names_by_prefix
from utils.trialbot.dummy_translator import install_dummy_translator
from utils.trialbot.setup_cli import setup as setup_cli
from utils.trialbot.setup_bot import setup_bot
from utils.s2s_arch.hparam_modifiers import param_overwriting_modifier, install_runtime_modifiers
from utils.lark.id_tree import build_from_lark_tree
from utils.tree import Tree, PreOrderTraverse, PostOrderTraverse


def main():
    install_semantic_parsing_datasets()

    install_dummy_translator()

    args = setup_cli(seed=2021, hparamset='moving-metric', **{"dry-run": None})

    install_runtime_modifiers(args.hparamset, partial(param_overwriting_modifier, **dict(
      zip(('src_key', 'tgt_key'), get_field_names_by_prefix(args.dataset))
    )))
    train, _, _ = Registry.get_dataset(args.dataset)

    bot = TrialBot(args, get_model_func=MetricModel.get_model)
    if not args.test:
        bot = setup_bot(bot, vdrop_reset=False, epoch_model_saving=False)
    else:
        bot = setup_bot(bot, False, False, False, True, False, False)

        @bot.attach_extension(Events.ITERATION_COMPLETED)
        def manual_gc(bot):
            import gc
            gc.collect()
            if bot.args.device >= 0:
                import torch.cuda
                torch.cuda.empty_cache()

    bot.run()


class MetricModel(torch.nn.Module):
    @classmethod
    def get_model(cls, p, vocab):
        return cls(p.chatglm_path, p.src_key, p.tgt_key, p.use_tgt_trees, p.max_tok_num)

    def __init__(self, llm_path: str, src_key: str, tgt_key: str,
                 use_tgt_trees: bool = True,
                 max_toks_num: int = 10000,
                 ):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True, revision='v0.1.0')
        self.model = AutoModel.from_pretrained(llm_path, trust_remote_code=True, revision='v0.1.0').half().eval()
        self.src_key = src_key
        self.tgt_key = tgt_key
        self.logger = logging.getLogger(self.__class__.__name__)
        from allennlp.training.metrics import Average
        self.avg_metric = Average()
        self.use_tgt_trees = use_tgt_trees
        self.max_toks_num = max_toks_num

    @torch.inference_mode()
    def forward(self, **kwargs):
        # list of str, or trees
        torch.no_grad()
        src = kwargs[self.src_key]

        if self.use_tgt_trees:
            tgt_trees = [self.compute_offsets(build_from_lark_tree(t))
                         for t in kwargs[self.tgt_key + '_tree'] if t is not None]
            if len(tgt_trees) == 0:
                return

            tgt = self.restore_text_from_trees(tgt_trees)
            offset_of_trees: list = [[node.payload for node in PreOrderTraverse()(t)]
                                     for t in tgt_trees]
        else:
            tgt = kwargs[self.tgt_key]
            offset_of_trees = []

        # (batch, len(s or t), dim), (batch, len(s or t))
        src_emb, src_ids = self.embed(src)
        tgt_emb, tgt_ids = self.embed(tgt)

        if self.use_tgt_trees:
            it = zip(src_emb, src_ids, tgt_emb, tgt_ids, offset_of_trees)
            for x_emb, x_ids, y_emb, y_ids, y_offsets in it:
                x_node_emb = self.compute_flat_seq_emb(x_emb, x_ids).double()
                y_node_emb = self.compute_tree_style_emb(y_ids, y_emb, y_offsets).double()
                self.compute_metric(x_node_emb, y_node_emb)

        else:
            it = zip(src_emb, src_ids, tgt_emb, tgt_ids)
            for x_emb, x_ids, y_emb, y_ids in it:
                x_node_emb = self.compute_flat_seq_emb(x_emb, x_ids).double()
                y_node_emb = self.compute_flat_seq_emb(y_emb, y_ids).double()
                self.compute_metric(x_node_emb, y_node_emb)

    def infer_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def embed(self, xs: list):
        xs_inputs = self.tok(xs,
                             padding=True,
                             return_tensors='pt').to(self.infer_device())
        xs_out = self.model(**xs_inputs, output_hidden_states=True)
        xs_emb = xs_out.hidden_states[-1].transpose(0, 1)
        xs_ids = xs_inputs['input_ids']
        return xs_emb, xs_ids   # (batch, len, dim)

    def compute_metric(self, x_node_emb, y_node_emb):
        """
        :param x_node_emb: (num1, dim)
        :param y_node_emb: (num2, dim)
        :return:
        """
        num_x_node, num_y_node = x_node_emb.size(0), y_node_emb.size(0)
        dev = x_node_emb.device

        Mf = ot.dist(x_node_emb, y_node_emb, metric='euclidean')

        x_uni = torch.ones((num_x_node,), device=dev, dtype=torch.float64) / num_x_node
        y_uni = torch.ones((num_y_node,), device=dev, dtype=torch.float64) / num_y_node

        dist = ot.emd2(x_uni, y_uni, Mf)

        del Mf, x_uni, y_uni, x_node_emb, y_node_emb
        self.logger.info(f'distance: {dist.item()}')
        self.avg_metric(dist)

    def compute_flat_seq_emb(self, emb, x_ids):
        """
        :param emb: (padded_tok_num, dim)
        :param x_ids: (padded_tok_num,)
        :return: (tok_num + 1, dim)
        """
        mask = (x_ids > 20004).unsqueeze(-1)   # printable tokens starts from 20005
        hid_sz = emb.size(-1)
        selected_embs = torch.masked_select(emb, mask).reshape(-1, hid_sz)[:self.max_toks_num]
        sent_emb = selected_embs.mean(0).unsqueeze(0)   # (1, dim)
        src_emb = torch.cat([selected_embs, sent_emb], dim=0)   # (sel_toks + 1, dim)
        del selected_embs, sent_emb, mask
        return src_emb

    def compute_tree_style_emb(self, term_ids, term_embs, tree_offsets):
        """
        Assuming a tree NT node is represented as the mean of the span.
        :param term_ids: (tok_num,)
        :param term_embs: (tok_num, dim)
        :param tree_offsets: list of spans with the range [start, end), including terminals
        :return: (num_nodes, dim)
        """
        len_contrib: list = [len(self.tok.decode(term_ids[:i+1]))
                             for i in range(len(term_ids))
                             if i < self.max_toks_num]

        def _find_tok_id(n, start=0) -> int:
            if n >= len_contrib[-1]:
                return -1

            for i, contrib in enumerate(len_contrib):
                if start <= n < contrib:
                    return i
                start = contrib

            # raise ValueError(f'invalid n={n} exceeds the input length {len_contrib[-1]}')
            return -1

        node_embs = []
        for start, end in tree_offsets:
            start_id = _find_tok_id(start)
            end_id = _find_tok_id(end - 1)  # end is included
            if start_id >= 0 and end_id >= 0:   # filter out the too long seq
                node_embs.append(term_embs[start_id:end_id + 1].mean(dim=0))    # (dim,)

        tgt_emb = torch.stack(node_embs, dim=0)
        del node_embs
        return tgt_emb  # (num_nodes, dim)

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

    @staticmethod
    def restore_text_from_trees(t_or_ts):
        def _restore(t):
            return ' '.join(node.label for node in PreOrderTraverse()(t) if node.is_terminal)

        if isinstance(t_or_ts, Tree):
            return _restore(t_or_ts)
        else:
            return [_restore(t) for t in t_or_ts]

    def get_metrics(self, reset: bool = False):
        return {"DIST": self.avg_metric.get_metric(reset=reset)}


@Registry.hparamset('moving-metric')
def base():
    from trialbot.training.hparamset import HyperParamSet
    from trialbot.utils.root_finder import find_root
    p = HyperParamSet.common_settings(find_root())
    p.chatglm_path = osp.expanduser('~/.glm/chatglm-6b')
    p.TRANSLATOR = 'dummy'
    p.TRANSLATOR_KWARGS = dict(filter_none=False)
    p.TRAINING_LIMIT = 1
    p.batch_sz = 4  # must be small, otherwise the GPU memory will goes run out.
    p.use_tgt_trees = True
    p.max_tok_num = 1100    # >num of 99% examples on ATIS
    return p


@Registry.hparamset('moving-metric-flat')
def plain_seq():
    p = base()
    p.use_tgt_trees = False
    return p


if __name__ == '__main__':
    main()
