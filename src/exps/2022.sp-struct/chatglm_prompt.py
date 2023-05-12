import json
import logging
from typing import Mapping, Generator, Tuple, List, Optional

import torch
from trialbot.data.translator import Translator, NullableTensor, FieldAwareTranslator
from trialbot.training import TrialBot, Registry
from trialbot.utils.root_finder import find_root
import os.path as osp
from transformers import AutoTokenizer, AutoModel


@Registry.hparamset('icl')
def chatglm():
    from trialbot.training.hparamset import HyperParamSet
    from trialbot.utils.root_finder import find_root
    p = HyperParamSet.common_settings(find_root())
    p.chatglm_path = osp.expanduser('~/.glm/chatglm-6b')
    p.TRANSLATOR = 'icl'
    p.prompt_mode = 'history'  # default, other choices are prompt

    # not useful, closed by default instead of removing relevant codes
    p.use_syntactic_prompt = False
    return p


@Registry.hparamset('icl_prompt')
def glm_pmpt():
    p = chatglm()
    p.prompt_mode = 'prompt'
    return p


@Registry.hparamset('ctx_dump')
def glm_dump():
    """do not call the model or downstream API, only dumps the prompt and context"""
    p = chatglm()
    p.prompt_mode = 'context_dump'
    return p


@Registry.hparamset('syn_prompt')
def glm_syn_prompt():
    p = glm_pmpt()
    p.use_syntactic_prompt = True
    return p


@Registry.hparamset('zero_shot')
def glm_zeroshot():
    p = chatglm()
    p.prompt_mode = 'zero_shot'
    return p


class WrapperModel(torch.nn.Module):
    def __init__(self, path, comp_fn=None, prompt_mode: str = 'history'):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, revision='v0.1.0')
        self.model = AutoModel.from_pretrained(path, trust_remote_code=True, revision='v0.1.0').half()
        self.model.eval()

        self.acc = 0
        self.count = 0
        self.comp_fn = comp_fn or (lambda x, y: x == y)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.prompt_mode = prompt_mode.lower()

    def forward(self, src, tgt, context):
        from utils.llm.prompt import build_prompt
        for x, y, c in zip(src, tgt, context):
            spec = {'src': x, 'tgt': y}

            if self.prompt_mode == 'history':
                assert len(c) == 0 or len(c[0]) == 2, "Context must be request-response pairs"
                y_hat = self.call_model(x, c)
                spec['ctx'] = c
                spec['pred'] = y_hat

            elif self.prompt_mode == 'prompt':
                prompt = build_prompt(x, c)
                y_hat = self.call_model(prompt, [])
                spec['prompt'] = prompt
                spec['pred'] = y_hat

            elif self.prompt_mode == 'context_dump':
                y_hat = y
                spec['ctx'] = c
                spec['prompt'] = build_prompt(x, c)

            elif self.prompt_mode == 'zero_shot':
                y_hat = self.call_model(x, [])
                spec['pred'] = y_hat

            else:
                raise ValueError('specified invalid prompt.')

            self.count += 1
            self.acc += 1 if self.comp_fn(y_hat, y) else 0
            self.logger.debug(json.dumps(spec))

    def call_model(self, user_input, history):
        res, _ = self.model.chat(self.tok, user_input, history=history)
        return res

    def get_metrics(self, reset: bool = False):
        m = {"ACC": round((self.acc / self.count) if self.count > 0 else 0, 4)}
        if reset:
            self.acc = self.count = 0
        return m

    @staticmethod
    def get_model(p, vocab):
        return WrapperModel(p.chatglm_path, prompt_mode=p.prompt_mode)


@Registry.translator('icl')
class ICLPromptTranslator(Translator):
    def batch_tensor(self, tensors: List[Mapping[str, NullableTensor]]):
        tensors = list(filter(lambda x: all(v is not None for v in x.values()), tensors))
        batch_dict = FieldAwareTranslator.list_of_dict_to_dict_of_list(tensors)
        return batch_dict

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        yield from []

    def to_tensor(self, example) -> Mapping[str, NullableTensor]:
        src, tgt = map(example.get, (self.src_field, self.tgt_field))
        if self.indexing_dataset is None:
            return {'src': src, 'tgt': tgt, 'context': []}

        exemplars: List[dict] = self.search_index(src)
        from utils.llm.prompt import build_ctx_with_exemplars
        ctx = build_ctx_with_exemplars(
            exemplars, self.src_field, self.tgt_field,
            use_ast='as_brackets' if self.use_syn else 'none',   # noqa
            ast_field=self.tgt_field + '_tree'
        )
        return {'src': src, 'tgt': tgt, 'context': ctx}

    def __init__(self):
        super().__init__()
        self.idx_conn = None
        self.src_field = None
        self.tgt_field = None
        self.tok = None
        self.indexing_dataset = None
        self.use_syn = False
        self.prompt_mode = None

    def lazy_init(self, bot: TrialBot):
        p = bot.hparams
        args = bot.args
        self.src_field, self.tgt_field = get_field_names(args.dataset)
        self.tok = AutoTokenizer.from_pretrained(p.chatglm_path,
                                                 trust_remote_code=True, revision='v0.1.0')
        self.use_syn: bool = getattr(p, 'use_syntactic_prompt', False)
        self.prompt_mode = p.prompt_mode
        if p.prompt_mode != 'zero_shot':
            self.indexing_dataset = bot.train_set
            self.load_index()

    def load_index(self):
        if self.idx_conn is not None:
            return

        from utils.ir_client import VolatileBM25Index
        self.idx_conn = VolatileBM25Index.from_data_list(
            keys=[x.get(self.src_field) for x in self.indexing_dataset],
            payloads=list(range(len(self.indexing_dataset)))
        )

    def search_index(self, key: str) -> List[dict]:
        self.load_index()
        return [self.indexing_dataset[i] for _, i in self.idx_conn.search_index(key)]


def main():
    from shujuji import cogs, compact_cfq as ccfq, smcalflow_cs as smc, cg_bundle as agsa, cofe
    smc.install()
    ccfq.install_dataset()
    cogs.install_dataset()
    agsa.install_parsed_qa_datasets()
    agsa.install_cross_domain_parsed_qa_datasets()
    cofe.install_datasets()

    from utils.trialbot.setup_cli import setup as setup_cli
    from utils.trialbot.setup_bot import setup_bot
    args = setup_cli(seed=2021, device=0, hparamset='icl', test=None)    # always test mode
    bot = TrialBot(args=args, trial_name='glm-6b', get_model_func=WrapperModel.get_model)
    bot.translator.lazy_init(bot)
    bot = setup_bot(bot, epoch_model_saving=False, epoch_test_eval=False, epoch_dev_eval=False,
                    metric_printing=True, vdrop_reset=False, use_gc=False)

    @bot.attach_extension()
    def report_metrics(bot, interval=5):
        from utils.trialbot.extensions import get_metrics
        if bot.state.iteration % interval == 0:
            get_metrics(bot, "Running Metrics:", reset=False)

    bot.run()


def get_field_names(ds_name: str):
    pref_confs = {
        'cogs': ('nl', 'lf'),
        'geo': ('sent', 'sql'),
        'ati': ('sent', 'sql'),
        'sch': ('sent', 'sql'),
        'adv': ('sent', 'sql'),
        'smc': ('utterance', 'plan'),
        'ccfq': ('source', 'target'),
        'cofe': ('context', 'ground_truth'),
    }
    for k, v in pref_confs.items():
        if ds_name.startswith(k):
            return v
    return None


if __name__ == '__main__':
    import sys
    sys.path.insert(0, find_root('.SRC'))
    main()
