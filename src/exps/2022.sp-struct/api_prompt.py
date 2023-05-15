import json
import sys
from typing import Any, Callable
import logging
import torch

from trialbot.training import Registry, TrialBot
from trialbot.utils.root_finder import find_root

sys.path.insert(0, find_root('.SRC'))

from utils.trialbot.setup_cli import setup as setup_cli
from utils.llm.openai_prompt import load_api_key_from_envion, completion, chat_completion
from utils.llm.prompt_translator import install_translator
from shujuji import (
    cogs,
    compact_cfq as ccfq,
    smcalflow_cs as smc,
    cg_bundle as agsa,
    cofe,
    get_field_names_by_prefix,
)


@Registry.hparamset('chat')
def openai_chat():
    from trialbot.training.hparamset import HyperParamSet
    from trialbot.utils.root_finder import find_root
    p = HyperParamSet.common_settings(find_root())
    p.TRANSLATOR = 'prompt'
    p.TRANSLATOR_KWARGS = dict(src_field=None, tgt_field=None, prompt_mode='prompt')
    p.use_chat = True
    return p


@Registry.hparamset('completion')
def openai_completion():
    p = openai_chat()
    p.use_chat = False
    return p


class WrapperModel(torch.nn.Module):
    def __init__(self, comp_fn: Callable[[Any, Any], bool] = None,
                 use_chat: bool = False, dry_run: bool = False):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.acc = 0
        self.count = 0
        self.comp_fn = comp_fn or (lambda x, y: x == y)
        self.use_chat = use_chat
        self.dry_run = dry_run

    def forward(self, src, tgt, context, prompt):
        for x, y, c, p in zip(src, tgt, context, prompt):
            spec = {'src': x, 'tgt': y, 'ctx': c, 'prompt': p}
            try:
                res = chat_completion(p) if self.use_chat else completion(p)
                success = True
            except Exception as e:
                res = ''
                success = False

            is_correct = self.comp_fn(res, y)
            spec.update(api_success=success, pred=res, num_processed=self.count, correct=is_correct)
            self.acc += 1 if is_correct else 0
            self.count += 1
            self.logger.debug(json.dumps(spec))

    def get_metrics(self, reset: bool = False):
        m = {"ACC": round((self.acc / self.count) if self.count > 0 else 0, 4)}
        if reset:
            self.count = self.acc = 0
        return m

    @classmethod
    def from_param_and_vocab(cls, param, vocab):
        return WrapperModel(use_chat=param.use_chat)


def main():
    smc.install()
    ccfq.install_dataset()
    cogs.install_dataset()
    agsa.install_parsed_qa_datasets()
    agsa.install_cross_domain_parsed_qa_datasets()
    cofe.install_datasets()
    install_translator()
    load_api_key_from_envion()

    args = setup_cli(seed=2021, device=-1, test=None)    # always test mode
    bot = TrialBot(args, 'llm_api', WrapperModel.from_param_and_vocab)
    trans: 'PromptTranslator' = bot.translator
    trans.src_field, trans.tgt_field = get_field_names_by_prefix(args.dataset)
    trans.load_index(bot.train_set)
    bot.run()


if __name__ == '__main__':
    main()
