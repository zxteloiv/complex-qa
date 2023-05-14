import json
import logging

import torch
from trialbot.training import TrialBot, Registry
from trialbot.utils.root_finder import find_root
import os.path as osp
from transformers import AutoTokenizer, AutoModel


PREF_CONFS = {
    'cogs': ('nl', 'lf'),
    'geo': ('sent', 'sql'),
    'ati': ('sent', 'sql'),
    'sch': ('sent', 'sql'),
    'adv': ('sent', 'sql'),
    'smc': ('utterance', 'plan'),
    'ccfq': ('source', 'target'),
    'cofe': ('context', 'ground_truth'),
}


@Registry.hparamset('icl_history')
def chatglm():
    from trialbot.training.hparamset import HyperParamSet
    from trialbot.utils.root_finder import find_root
    p = HyperParamSet.common_settings(find_root())
    p.chatglm_path = osp.expanduser('~/.glm/chatglm-6b')
    p.TRANSLATOR = 'prompt'
    p.TRANSLATOR_KWARGS = dict(src_field=None, tgt_field=None, prompt_mode=None)
    p.prompt_mode = 'history'
    p.batch_sz = 16
    p.dry_run = False
    return p


@Registry.hparamset('prompt')
def glm_pmpt():
    p = chatglm()
    p.prompt_mode = 'prompt'
    return p


@Registry.hparamset('dry_run')
def glm_dump():
    """do not call the model or downstream API, only dumps the prompt and context"""
    p = chatglm()
    p.prompt_mode = 'prompt'
    p.dry_run = True
    return p


@Registry.hparamset('syn_prompt')
def glm_syn_prompt():
    p = glm_pmpt()
    p.prompt_mode = 'syn_prompt'
    return p


@Registry.hparamset('zero_shot')
def glm_zeroshot():
    p = chatglm()
    p.prompt_mode = 'zero_shot'
    return p


class WrapperModel(torch.nn.Module):
    def __init__(self, path, comp_fn=None, use_history: bool = True, dry_run: bool = False):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, revision='v0.1.0')
        self.model = AutoModel.from_pretrained(path, trust_remote_code=True, revision='v0.1.0').half()
        self.model.eval()

        self.acc = 0
        self.count = 0
        self.comp_fn = comp_fn or (lambda x, y: x == y)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.use_history = use_history  # use history or in-line prompt
        self.dry_run = dry_run

    def forward(self, src, tgt, context, prompt):
        for x, y, c, p in zip(src, tgt, context, prompt):
            spec = {'src': x, 'tgt': y, 'ctx': c, 'prompt': p}
            if self.dry_run:
                y_hat = y
            elif self.use_history:
                y_hat, _ = self.model.chat(self.tok, x, history=c)
            else:
                y_hat, _ = self.model.chat(self.tok, p, history=[])

            spec['pred'] = y_hat
            self.count += 1
            self.acc += 1 if self.comp_fn(y_hat, y) else 0
            self.logger.debug(json.dumps(spec))

    def get_metrics(self, reset: bool = False):
        m = {"ACC": round((self.acc / self.count) if self.count > 0 else 0, 4)}
        if reset:
            self.acc = self.count = 0
        return m

    @staticmethod
    def get_model(p, vocab):
        return WrapperModel(p.chatglm_path,
                            use_history=p.prompt_mode == 'history', dry_run=p.dry_run)


def main():
    from shujuji import cogs, compact_cfq as ccfq, smcalflow_cs as smc, cg_bundle as agsa, cofe
    smc.install()
    ccfq.install_dataset()
    cogs.install_dataset()
    agsa.install_parsed_qa_datasets()
    agsa.install_cross_domain_parsed_qa_datasets()
    cofe.install_datasets()

    from utils.llm.prompt_translator import install_translator
    install_translator()

    from utils.trialbot.setup_cli import setup as setup_cli
    from utils.trialbot.setup_bot import setup_bot
    args = setup_cli(seed=2021, device=0, test=None)    # always in test mode
    bot = TrialBot(args=args, trial_name='glm-6b', get_model_func=WrapperModel.get_model)
    lazy_init_translator(bot)
    bot = setup_bot(bot, epoch_model_saving=False, epoch_test_eval=False, epoch_dev_eval=False,
                    metric_printing=True, vdrop_reset=False, use_gc=False)

    @bot.attach_extension()
    def report_metrics(bot, interval=5):
        from utils.trialbot.extensions import get_metrics
        if bot.state.iteration % interval == 0:
            get_metrics(bot, "Running Metrics:", reset=False)

    bot.run()


def lazy_init_translator(bot):
    args, p = bot.args, bot.hparams
    trans = bot.translator
    trans.src_field, trans.tgt_field = get_field_names(args.dataset)
    trans.prompt_mode = p.prompt_mode
    if p.prompt_mode != 'zero_shot':
        trans.load_index(bot.train_set)


def get_field_names(ds_name: str):
    for k, v in PREF_CONFS.items():
        if ds_name.startswith(k):
            return v
    return None


if __name__ == '__main__':
    import sys
    sys.path.insert(0, find_root('.SRC'))
    main()
