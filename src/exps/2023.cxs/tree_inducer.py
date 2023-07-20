from trialbot.data.fields import SeqField
from trialbot.utils import prepend_pythonpath   # noqa
from trialbot.data.translator import FieldAwareTranslator
from typing import Any, TypeVar, cast
import torch.nn

import os.path as osp
import trialbot.data
from trialbot.training import TrialBot, Registry
from trialbot.utils.root_finder import find_root
from trialbot.training.hparamset import HyperParamSet
from allennlp.training.metrics import Average
from allennlp.common.util import int_to_device
from functools import partial

from models.base_s2s.base_seq2seq import BaseSeq2Seq
from models.interfaces.token_predictor import TokenPredictor
from models.modules.token_predictors import MoSPredictor
from models.onlstm.stacked_onlstm import StackedONLSTM
from shujuji import install_semantic_parsing_datasets, get_field_names_by_prefix
from models.base_s2s.model_factory import Seq2SeqBuilder
from utils.nn import seq_cross_ent
from utils.s2s_arch.translators import AutoPLMField
from utils.seq_collector import SeqCollector
from utils.trialbot.dummy_translator import DummyField
from utils.trialbot.setup_cli import setup as setup_cli
from utils.s2s_arch.hparam_modifiers import (
    install_runtime_modifiers, overwriting_mod_template, MODIFIER, hp_prefix_match
)
from utils.llm.tokenizer_adapter import get_llm_wrapper_split_fn
from tree_prior import TreeInducer, PriorAgent
from tree_prior import install_hparams as install_prior_hparams
from utils.trialbot.setup_bot import setup_bot

T = torch.Tensor


def main():
    install_semantic_parsing_datasets()
    install_hparams()
    install_translators()

    parser = TrialBot.get_default_parser()
    parser.add_argument('--prior', help='the prior model file name')
    args = setup_cli(parser, seed=2021, device=-1)
    install_runtime_modifiers(args.hparamset, get_runtime_modifiers(args))

    agent = None if not args.prior else load_prior_agent(args.prior, device=args.device)

    bot = TrialBot(args, trial_name=args.hparamset,
                   get_model_func=get_model_func(args.hparamset, agent))
    bot = setup_bot(bot, True, True, False, True, True, True)
    bot.run()


def load_prior_agent(model_path: str, hparamset: str = 'base-prior', device: int = -1):
    install_prior_hparams()
    p = Registry.get_hparamset(hparamset)
    inducer = TreeInducer.new(Registry.get_hparamset(hparamset), None)
    inducer.load_state_dict(torch.load(model_path, int_to_device(device)))
    if device >= 0:
        inducer.cuda(device)
    agent = PriorAgent(inducer, p.llm_path, device, p.num_tree_sampling)
    return agent


def install_hparams():
    @Registry.hparamset()
    def y_inducer():
        p = HyperParamSet.common_settings(find_root())
        p.TRAINING_LIMIT = 150
        p.batch_sz = 16
        p.tgt_namespace = 'target_tokens'
        p.emb_sz = 100
        p.hid_sz = 320
        p.chunk_sz = 10
        p.dropout = 0.1
        p.n_mixtures = 5

        p.llm_path = osp.expanduser('~/.glm/chatglm-6b')
        return p

    @Registry.hparamset()
    def yx_inducer():
        p = Seq2SeqBuilder.base_hparams()
        p.TRAINING_LIMIT = 150
        p.batch_sz = 16

        p.hidden_sz = 320
        p.dec_in_dim = p.hidden_sz  # by default
        p.dec_out_dim = p.hidden_sz  # by default
        p.proj_in_dim = p.hidden_sz  # by default
        p.dec_onlstm_chunk_sz = 10  # such that there are 32 chunks in the onlstm internal

        p.src_namespace = None
        p.decoder_init_strategy = "avg_all"
        p.lr_scheduler_kwargs = None
        p.plm_name = osp.expanduser('~/.cache/manual_llm_cache/bert-base-uncased')
        p.encoder = f'plm:{p.plm_name}'
        p.decoder = 'onlstm'
        p.num_dec_layers = 3

        p.src_emb_pretrained_file = None    # the embedding-based

        p.llm_path = osp.expanduser('~/.glm/chatglm-6b')
        return p

    @Registry.hparamset()
    def yx_inducer_with_prior():
        p = yx_inducer()
        p.batch_sz = 8  # if we put ChatGLM on GPU the batch must be sufficiently small
        return p


def install_translators():
    @Registry.translator('y_inducer')
    class YTranslator(FieldAwareTranslator):
        def __init__(self, tgt_key: str, llm_path: str):
            super().__init__(field_list=[SeqField(
                tgt_key,
                renamed_key='ys_token',
                split_fn=get_llm_wrapper_split_fn(llm_path=llm_path),
                max_seq_len=1100,
            )])

    @Registry.translator('yx_inducer')
    class YXTranslator(FieldAwareTranslator):
        def __init__(self, src_key: str, tgt_key: str, src_llm: str, tgt_llm: str):
            super().__init__(field_list=[
                AutoPLMField(src_key, src_llm, 'source_tokens', trust_remote_code=True),
                SeqField(tgt_key, 'target_tokens', split_fn=get_llm_wrapper_split_fn(tgt_llm),
                         max_seq_len=1100),
            ])

    @Registry.translator('yx_inducer_with_prior')
    class YXTranslator(FieldAwareTranslator):
        def __init__(self, src_key: str, tgt_key: str, src_llm: str, tgt_llm: str):
            super().__init__(field_list=[
                AutoPLMField(src_key, src_llm, 'source_tokens', trust_remote_code=True),
                SeqField(tgt_key, 'target_tokens', split_fn=get_llm_wrapper_split_fn(tgt_llm),
                         max_seq_len=1100),
                DummyField([tgt_key], ['target_raw_str']),
            ])


def get_runtime_modifiers(args) -> list[MODIFIER]:
    # infer hparams based on runtime args
    ds, hp = args.dataset, args.hparamset

    key_nl, key_lf = get_field_names_by_prefix(ds)
    # training limit
    limit_conf: dict[str, int] = {
        'geo': 200, 'sch': 200, 'ati': 40, 'adv': 40,
        'cogs': 15, 'ccfq': 15, 'cfq': 15, 'smc': 15, 'cofe': 15,
    }

    hp_ow_mod = partial(overwriting_mod_template,
                        TRAINING_LIMIT=hp_prefix_match(limit_conf, ds),  # training limit varies across dbs
                        TRANSLATOR=hp,         # translators are registered in the same name
                        src_namespace=key_nl,
                        tgt_namespace=key_lf,)

    def hp_lazy_mod(p: HyperParamSet):
        if hp == 'y_inducer':
            p.TRANSLATOR_KWARGS = {"tgt_key": p.tgt_namespace, "llm_path": p.llm_path}

        elif hp.lower().startswith('yx_inducer'):
            # the two translators use the same parameter definitions.
            p.TRANSLATOR_KWARGS = dict(src_key=p.src_namespace, tgt_key=p.tgt_namespace,
                                       src_llm=p.plm_name, tgt_llm=p.llm_path,)
        return p

    return [hp_ow_mod, hp_lazy_mod]


def get_model_func(hp: str, prior_agent=None):
    if hp.lower() == 'y_inducer':
        return YInducer.new
    elif hp.lower() == 'yx_inducer':
        return YXInducer.new
    elif hp.lower() == 'yx_inducer_with_prior':
        assert prior_agent is not None, 'specified prior setting but no prior is given.'
        return partial(YXInducer.new, prior_agent=prior_agent)

    raise NotImplementedError


class YInducer(torch.nn.Module):
    """
    The model used to learn and induce structures based on solely the y's. P(y)
    """
    @classmethod
    def new(cls, p, vocab: trialbot.data.NSVocabulary):
        return cls(
            embedding=torch.nn.Embedding(vocab.get_vocab_size(p.tgt_namespace), p.emb_sz, padding_idx=0),
            rnn=StackedONLSTM(p.emb_sz, p.hid_sz, p.chunk_sz, 3, p.dropout),    # fixed with 3 layers
            token_predictor=MoSPredictor(p.hid_sz, vocab.get_vocab_size(p.tgt_namespace), p.n_mixtures),
        )

    def __init__(self,
                 embedding: torch.nn.Embedding,
                 rnn: StackedONLSTM,
                 token_predictor: TokenPredictor,
                 padding: int = 0,
                 s_layer: int = 1,  # induced structure is at the s_layer
                 ):
        super().__init__()
        self.embedding = embedding
        self.rnn = rnn
        self.predictor = token_predictor
        self.padding = padding
        self.s_layer: int = s_layer
        self.acc = Average()

    def forward(self, ys_token: T):
        # ys_token: (batch, n)
        ys_in = ys_token[:, :-1].contiguous()   # (batch, n-1)
        ys_emb = self.embedding(ys_in)   # (b, n-1, d)

        # assuming the token are all at the right
        hx = None
        step_mem = SeqCollector()
        for t in range(ys_emb.size(1)):
            emb_t = ys_emb[:, t]    # (b, d)
            hx, o = self.rnn(emb_t, hx)
            df = hx[self.s_layer][-1]  # (b, #chunk)
            cf = hx[self.s_layer][2]    # (b,)
            step_mem(out=o, df=df, cf=cf)

        rnn_out = step_mem.get_stacked_tensor('out')    # (b, n-1, d)
        logits = self.predictor(rnn_out)    # (b, n-1, v)

        gold = ys_token[:, 1:].contiguous()  # (b, n-1)
        mask: T = gold != self.padding
        loss = seq_cross_ent(logits, gold, mask.float())

        pred: T = logits.argmax(dim=-1)  # (b, n-1)
        correct = torch.logical_or((pred == gold), ~mask)  # (b, n-1)
        length = correct.size(1)

        for sample in correct:
            self.acc(sample.sum() == length)

        output = {'loss': loss,     # (,), scalar
                  'df': step_mem.get_stacked_tensor('df'),    # (b, n-1, #chunk)
                  'cf': step_mem.get_stacked_tensor('cf'),    # (b, n-1)
                  'ys': ys_token,   # (b, n)
                  'pred': pred,     # (b, n-1)
                  }
        return output

    def get_metrics(self, reset: bool = False) -> dict[str, float]:
        acc = self.acc.get_metric(reset=reset)
        return dict(ACC=round(acc, 4))


class YXInducer(BaseSeq2Seq):
    @classmethod
    def new(cls, p, vocab, prior_agent=None) -> 'YXInducer':
        from models.base_s2s.model_factory import Seq2SeqBuilder
        model = cast('YXInducer', Seq2SeqBuilder(p, vocab).get_model(cls))
        setattr(model, 's_layer', 1)
        if prior_agent is not None:
            setattr(model, 'prior', prior_agent)
        return model

    def forward(self,
                source_tokens: torch.LongTensor | dict,
                target_tokens: torch.LongTensor = None,
                **kwargs,
                ) -> dict[str, torch.Tensor]:
        self._reset_variational_dropouts()

        layer_states, state_mask = self._embed_encoder(source_tokens)
        hx, enc_attn_fn, start = self._prepare_dec(layer_states, state_mask.long())
        preds, logits, df, cf = self._forward_dec(target_tokens, start, enc_attn_fn, hx)

        if not isinstance(source_tokens, torch.Tensor):
            source_tokens = source_tokens['input_ids']  # if the source token is a UserDict for PLM models

        output = {'source': source_tokens, "loss": 0}
        if self.training:
            output['loss'] = self._compute_loss(logits, target_tokens, state_mask)

        if target_tokens is not None:
            total_err = self._compute_metrics(source_tokens, target_tokens, preds, logits)
            target_mask = (target_tokens[:, 1:] != self._padding_index)
            self._compute_gini_index(state_mask, target_mask)
            output.update(errno=total_err.tolist())

        output.update(pred=preds, ys=target_tokens, df=df, cf=cf)
        self.prior_loss(cast(list[str], kwargs.get('target_raw_str')), df)
        return output

    def _forward_dec(self, target_tokens, default_start, enc_attn_fn, hx):
        target, target_mask = target_tokens, (target_tokens != self._padding_index)
        # preds: (batch, seq_len)
        # logits: (batch, seq_len, vocab_size)
        # preds, logits = self._forward_dec_loop(start, enc_attn_fn, hx, target, target_mask, runtime_len)
        last_pred = default_start
        num_decoding_steps = self._get_decoding_loop_len(target)
        mem = SeqCollector()
        for timestep in range(num_decoding_steps):
            step_input = self._choose_rnn_input(last_pred, None if target is None else target[:, timestep])
            dec_hist_attn_fn = self._create_runtime_dec_hist_attn_fn(mem, target_mask, timestep)
            cell_out, step_logit, hx = self._forward_dec_loop(step_input, enc_attn_fn, dec_hist_attn_fn, hx)
            # last_pred: (batch,), greedy decoding
            last_pred = torch.argmax(step_logit, dim=-1)
            df = hx[self.s_layer][-1]  # (b, #chunk)
            cf = hx[self.s_layer][2]    # (b,)
            mem(output=cell_out, logit=step_logit, df=df, cf=cf)

        # logits: (batch, seq_len, vocab_size)
        # predictions: (batch, seq_len)
        logits = mem.get_stacked_tensor('logit')
        df = mem.get_stacked_tensor('df')   # (b, seq-len, #chunk)
        cf = mem.get_stacked_tensor('cf')   # (b, seq-len)
        predictions = logits.argmax(dim=-1)

        return predictions, logits, df, cf

    def prior_loss(self, ys_str: list[str], df):
        if getattr(self, 'prior', None) is None:
            return

        prior = cast(PriorAgent, self.prior)
        ys_emb, ys_id = prior.embed(ys_str)
        df_mask = prior.get_mask(ys_id)
        prior_logp_df = prior.inducer.get_batch_tree_dist(ys_emb, df_mask)  # (b, #seq, #chunk)
        post_logp_df = df.log_softmax(dim=-1)
        print('df sizes:', prior_logp_df.size(), post_logp_df.size())


if __name__ == '__main__':
    main()
