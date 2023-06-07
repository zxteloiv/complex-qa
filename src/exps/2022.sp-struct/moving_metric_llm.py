from functools import partial
import sys
from trialbot.utils.root_finder import find_root
import logging

sys.path.insert(0, find_root('.SRC'))

import torch
from transformers import AutoModel, AutoTokenizer
from trialbot.training import Registry, TrialBot
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
    bot = setup_bot(bot, vdrop_reset=False, epoch_model_saving=False)
    bot.run()


class MetricModel(torch.nn.Module):
    @classmethod
    def get_model(cls, p, vocab):
        return cls(p.chatglm_path, p.src_key, p.tgt_key)

    def __init__(self, llm_path: str, src_key: str, tgt_key: str):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True, revision='v0.1.0')
        self.model = AutoModel.from_pretrained(llm_path, trust_remote_code=True, revision='v0.1.0').half().eval()
        self.src_key = src_key
        self.tgt_key = tgt_key
        self.logger = logging.getLogger(self.__class__.__name__)
        from allennlp.training.metrics import Average
        self.avg_metric = Average()

    @torch.inference_mode()
    def forward(self, **kwargs):
        # list of str, or trees
        torch.no_grad()
        src = kwargs[self.src_key]
        tgt_trees = [self.compute_offsets(build_from_lark_tree(t))
                     for t in kwargs[self.tgt_key + '_tree'] if t is not None]
        if len(tgt_trees) == 0:
            return

        tgt = self.restore_text_from_trees(tgt_trees)
        offset_of_trees: list = [[node.payload for node in PreOrderTraverse()(t)] for t in tgt_trees]

        # (batch, len(s or t), dim), (batch, len(s or t))
        src_emb, src_ids = self.embed(src)
        tgt_emb, tgt_ids = self.embed(tgt)

        self.compute_metric(src_emb, src_ids, tgt_emb, tgt_ids, offset_of_trees)

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

    def compute_metric(self, src_emb, src_ids, tgt_emb, tgt_ids, tgt_offsets):
        """
        :param src_emb: (batch, tok_num, dim)
        :param src_ids: (batch, tok_num)
        :param tgt_emb: (batch, num_nodes, dim)
        :param tgt_ids: (batch, num_nodes)
        :param tgt_offsets: list of list of span
        :return:
        """
        it = zip(src_emb, src_ids, tgt_emb, tgt_ids, tgt_offsets)
        dev = src_emb.device
        for x_emb, x_ids, y_emb, y_ids, y_offsets in it:
            x_node_emb = self.compute_src_node_emb(x_emb, x_ids).double()
            y_node_emb = self.compute_tgt_node_emb(y_ids, y_emb, y_offsets).double()
            num_x_node, num_y_node = x_node_emb.size(0), y_node_emb.size(0)

            Mf = ot.dist(x_node_emb, y_node_emb, metric='euclidean')

            x_uni = torch.ones((num_x_node,), device=dev, dtype=torch.float64) / num_x_node
            y_uni = torch.ones((num_y_node,), device=dev, dtype=torch.float64) / num_y_node

            dist = ot.emd2(x_uni, y_uni, Mf)

            del Mf, x_uni, y_uni, x_node_emb, y_node_emb
            self.logger.info(f'distance: {dist.item()}')
            self.avg_metric(dist)

    def compute_src_node_emb(self, emb, x_ids):
        """
        :param emb: (padded_tok_num, dim)
        :param x_ids: (padded_tok_num,)
        :return: (tok_num + 1, dim)
        """
        required_mask = (x_ids > 20004).unsqueeze(-1)   # printable tokens starts from 20005
        hid_sz = emb.size(-1)
        selected_embs = torch.masked_select(emb, required_mask).reshape(-1, hid_sz)
        sent_emb = selected_embs.mean(0).unsqueeze(0)   # (1, dim)
        src_emb = torch.cat([selected_embs, sent_emb], dim=0)   # (sel_toks + 1, dim)
        del selected_embs, sent_emb, required_mask
        return src_emb

    def compute_tgt_node_emb(self, term_ids, term_embs, tree_offsets):
        """
        :param term_ids: (tok_num,)
        :param term_embs: (tok_num, dim)
        :param tree_offsets: list of spans with the range [start, end)
        :return: (num_nodes, dim)
        """
        len_contrib: list = [len(self.tok.decode(term_ids[:i+1])) for i in range(len(term_ids))]

        def _find_tok_id(n, start=0) -> int:
            assert n < len_contrib[-1]
            for i, contrib in enumerate(len_contrib):
                if start <= n < contrib:
                    return i
                start = contrib
            raise ValueError(f'invalid n={n} exceeds the input length {len_contrib[-1]}')

        node_embs = []
        for start, end in tree_offsets:
            start_id = _find_tok_id(start)
            end_id = _find_tok_id(end - 1)  # end is included
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
    return p


if __name__ == '__main__':
    main()
