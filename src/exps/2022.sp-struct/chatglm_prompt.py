import json
import sqlite3
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
    return p


@Registry.hparamset('icl_prompt')
def glm_pmpt():
    p = chatglm()
    p.prompt_mode = 'prompt'
    return p


@Registry.hparamset('ctx_dump')
def glm_dump():
    p = chatglm()
    p.prompt_mode = 'context_dump'
    return p


@Registry.hparamset('icl_syn')
def glm_syn_prompt():
    p = chatglm()
    p.prompt_mode = 'syntactic'
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
        for x, y, c in zip(src, tgt, context):
            spec = {'src': x, 'tgt': y}

            if self.prompt_mode == 'history':
                y_hat, _ = self.model.chat(self.tok, x, history=c)
                spec['pred'] = y_hat
            elif self.prompt_mode == 'prompt':
                y_hat, _ = self.model.chat(self.tok, self.build_prompt(x, c), history=[])
                spec['pred'] = y_hat
            elif self.prompt_mode == 'syntactic':
                pass
            elif self.prompt_mode == 'context_dump':
                y_hat = y
                spec['ctx'] = c
            else:
                raise ValueError('specified invalid prompt.')

            self.count += 1
            self.acc += 1 if self.comp_fn(y_hat, y) else 0
            self.logger.debug(json.dumps(spec))

    def build_prompt(self, x, ctx):
        template = "Input: {0}\nOutput: {1}"
        prompt = '\n'.join(template.format(*pair) for pair in ctx)
        prompt += f'\nInput: {x}\nOutput: '
        return prompt

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
        data = self.search_index(src)
        return {'src': src, 'tgt': tgt, 'context': data}

    def __init__(self):
        super().__init__()
        self.idx_conn = None
        self.src_field = None
        self.tgt_field = None
        self.tok = None
        self.indexing_dataset = None

    def lazy_init(self, bot: TrialBot):
        p = bot.hparams
        args = bot.args
        self.src_field, self.tgt_field = get_field_names(args.dataset)
        self.tok = AutoTokenizer.from_pretrained(p.chatglm_path,
                                                 trust_remote_code=True, revision='v0.1.0')
        self.indexing_dataset = bot.train_set
        self.load_index()

    def load_index(self):
        if self.idx_conn is not None:
            return

        conn = sqlite3.connect(':memory:')
        logging.debug('building index...')
        conn.execute('create virtual table kvmem using fts5(key, val);')
        conn.commit()
        data = [(x.get(self.src_field), x.get(self.tgt_field)) for x in self.indexing_dataset]
        logging.debug('inserting %d items into index...' % len(data))
        conn.executemany('insert into kvmem values (?, ?);', data)
        conn.commit()
        self.idx_conn = conn

    def search_index(self, key) -> List[Tuple]:
        self.load_index()

        keywords = []
        for k in set(key.split()):
            k = k.replace('"', '')
            keywords.append(f'"{k}"')

        fts_str = ' OR '.join(keywords)
        cur = self.idx_conn.execute(
            'select `key`, val from kvmem where `key` match (?) order by bm25(kvmem) limit 10',
            (fts_str,)
        )
        data = cur.fetchall()
        return data


def main():
    from shujuji import cogs, compact_cfq as ccfq, smcalflow_cs as smc, cg_bundle as agsa
    smc.install()
    ccfq.install_dataset()
    cogs.install_dataset()
    agsa.install_parsed_qa_datasets()
    agsa.install_cross_domain_parsed_qa_datasets()

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
    }
    for k, v in pref_confs.items():
        if ds_name.startswith(k):
            return v
    return None


if __name__ == '__main__':
    import sys
    sys.path.insert(0, find_root('.SRC'))
    main()
