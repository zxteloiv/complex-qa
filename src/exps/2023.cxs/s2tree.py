from collections.abc import Iterator

import torch
import trialbot.utils.prepend_pythonpath    # noqa
import os.path as osp

from trialbot.data.field import Field, T
from trialbot.data.translator import FieldAwareTranslator
from trialbot.training import Registry, TrialBot

from shujuji import install_semantic_parsing_datasets
from shujuji.tree_extending import install_extended_ds
from utils.preprocessing import nested_list_numbers_to_tensors
from utils.tree import Tree
from utils.trialbot.setup_cli import setup as setup_cli
from utils.s2s_arch.setup_bot import setup_common_bot as setup_bot
from utils.s2s_arch.translators.plm2s import AutoPLMField
from models.base_s2s.model_factory import Seq2SeqBuilder
from trialbot.data import START_SYMBOL, END_SYMBOL
from itertools import product


def main():
    install_semantic_parsing_datasets()
    install_extended_ds()
    install_hparamset()
    install_translators()

    args = setup_cli(seed=2021, device=0, hparamset='geo_cg', dataset='geo_cg_tree_yxp')
    bot = setup_bot(args, trialname=args.hparamset)
    bot.run()


def install_hparamset():
    @Registry.hparamset('geo_cg')
    def base():
        p = Seq2SeqBuilder.base_hparams()
        p.OPTIM = 'adabelief'
        p.TRAINING_LIMIT = 150
        p.batch_sz = 16

        p.src_namespace = None
        p.tgt_namespace = 'target_tokens'
        p.decoder_init_strategy = "avg_all"
        p.lr_scheduler_kwargs = None
        p.plm_name = osp.expanduser('~/.cache/manual_llm_cache/bert-base-uncased')
        p.encoder = f'plm:{p.plm_name}'
        p.TRANSLATOR = 'plm2tree'
        p.TRANSLATOR_KWARGS = dict(x_key='sent', y_key='induced_tgt_tree', x_plm=p.plm_name)

        p.src_emb_pretrained_file = None    # the embedding-based
        return p


def install_translators():
    @Registry.translator('plm2tree')
    class PLM2Tree(FieldAwareTranslator):
        def __init__(self, x_key, y_key, x_plm,):
            super().__init__(field_list=[
                AutoPLMField(x_key, x_plm, 'source_tokens'),
                TreeField(y_key, 'target_tokens', 'target_tokens')
            ])


class TreeField(Field):
    def build_batch_by_key(self, input_dict: dict[str, list[T]]) -> dict[str, torch.Tensor | list[T]]:
        input_list = input_dict.get(self.renamed_key)
        tensor = nested_list_numbers_to_tensors(input_list)
        return {self.renamed_key: tensor}

    def generate_namespace_tokens(self, example) -> Iterator[tuple[str, str]]:
        tree: Tree | None = example.get(self.source_key)
        if self.add_start_end_toks:
            yield from product([self.ns], [START_SYMBOL, END_SYMBOL])

        if tree is not None:
            for n in tree.iter_subtrees_topdown():
                yield self.ns, n.immediate_str()

    def to_input(self, example) -> dict[str, T | None]:
        tree: None | Tree = example.get(self.source_key)
        if tree is None:
            return {self.renamed_key: None}

        toks = [n.immediate_str() for n in tree.iter_subtrees_topdown()]

        if self.max_seq_len > 0:
            toks = toks[:self.max_seq_len]

        if self.add_start_end_toks:
            toks = [START_SYMBOL] + toks + [END_SYMBOL]

        toks = [self.vocab.get_token_index(t, self.ns) for t in toks]
        return {self.renamed_key: toks}

    def __init__(self, source_key: str,
                 renamed_key: str = None,
                 namespace: str = None,
                 add_start_end_toks: bool = True,
                 max_seq_len: int = 0,
                 ):
        super().__init__()
        self.source_key = source_key
        self.ns = namespace or source_key
        self.renamed_key = renamed_key or source_key
        self.add_start_end_toks = add_start_end_toks
        self.max_seq_len = max_seq_len


if __name__ == '__main__':
    main()
