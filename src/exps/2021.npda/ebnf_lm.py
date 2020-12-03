import sys
import os.path
sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))
import json

from trialbot.training import TrialBot
from trialbot.training import Registry
from utils.root_finder import find_root
from utils.trialbot_setup import setup
from trialbot.utils.move_to_device import move_to_device
from trialbot.data import NSVocabulary, PADDING_TOKEN, START_SYMBOL, END_SYMBOL
from torch import nn

import datasets.cfq
import datasets.cfq_translator

@Registry.hparamset()
def cfq_pattern():
    from trialbot.training.hparamset import HyperParamSet
    p = HyperParamSet.common_settings(find_root())
    p.ns = datasets.cfq_translator.PARSE_TREE_NS
    p.ns_fi = datasets.cfq_translator.NS_FI
    p.NS_VOCAB_KWARGS = {"non_padded_namespaces": p.ns[1:]}
    p.TRAINING_LIMIT = 1000
    p.emb_sz = 64
    p.hidden_dim = 64
    p.num_expander_layer = 2
    p.dropout = 0.2
    p.batch_sz = 32
    p.stack_capacity = 100
    p.ADAM_LR = 1e-3    # by default

    # When using a tied embedding, the projection-based quant is conflicted with max_norm embedding,
    # which is an inplace operation and modifies the graph.
    p.tied_nonterminal_emb = True
    p.tied_terminal_emb = True
    p.embedding_norm = False
    p.nt_emb_max_norm = None   # Optional[int], set to None rather than 0 to disable max_norm
    p.t_emb_max_norm = None
    p.nt_pred_crit = "projection" # projection based or distance based
    p.t_pred_crit = "projection"

    p.grammar_entry = 'queryunit'
    p.weight_decay = 0.2

    return p

@Registry.hparamset()
def cfq_pattern_both_norm():
    p = cfq_pattern()
    p.embedding_norm = True
    return p

@Registry.hparamset()
def cfq_finetune():
    p = cfq_pattern()
    p.TRAINING_LIMIT = 5
    p.ADAM_LR = 1e-5
    return p

def get_ebnf_lm(p, vocab: NSVocabulary):
    from models.neural_pda.ebnf_npda import NeuralEBNF
    from models.neural_pda.batched_stack import TensorBatchStack
    from models.neural_pda.formal_language_model import EBNFTreeLM
    from models.modules.stacked_rnn_cell import StackedLSTMCell
    from models.modules.quantized_token_predictor import QuantTokenPredictor
    from models.modules.normalization import Normalization

    ns_nt, ns_t, ns_et = p.ns
    emb_nt = nn.Embedding(vocab.get_vocab_size(ns_nt), p.emb_sz, max_norm=p.nt_emb_max_norm)
    emb_t = nn.Embedding(vocab.get_vocab_size(ns_t), p.emb_sz, max_norm=p.t_emb_max_norm)

    nt_pred = QuantTokenPredictor(
        num_toks=vocab.get_vocab_size(ns_nt),
        tok_dim=p.emb_sz,
        shared_embedding=emb_nt.weight if p.tied_nonterminal_emb else None,
        quant_criterion=p.nt_pred_crit,
    )
    t_pred = QuantTokenPredictor(
        num_toks=vocab.get_vocab_size(ns_t),
        tok_dim=p.emb_sz,
        shared_embedding=emb_t.weight if p.tied_terminal_emb else None,
        quant_criterion=p.t_pred_crit,
    )

    if p.embedding_norm:
        emb_nt = nn.Sequential(emb_nt, Normalization())
        emb_t = nn.Sequential(emb_t, Normalization())

    pda = NeuralEBNF(
        emb_nonterminals=emb_nt,
        emb_terminals=emb_t,
        num_nonterminals=vocab.get_token_index(ns_nt),
        ebnf_expander=StackedLSTMCell(
            input_dim=p.emb_sz * 2 + 1,
            hidden_dim=p.hidden_dim + 1,    # quant predictor requires input hidden == embedding size
            n_layers=p.num_expander_layer,
            intermediate_dropout=p.dropout,
        ),
        state_transition=None,
        batch_stack=TensorBatchStack(p.batch_sz, p.stack_capacity, 1 + 1),
        predictor_nonterminals=nt_pred,
        predictor_terminals=t_pred,
        start_token_id=vocab.get_token_index(START_SYMBOL, ns_nt),
        ebnf_entrypoint=vocab.get_token_index(p.grammar_entry, ns_nt),
        dropout=p.dropout,
    )

    model = EBNFTreeLM(pda, tok_pad_id=vocab.get_token_index(PADDING_TOKEN, ns_nt), nt_fi=p.ns_fi[0])
    return model

from trialbot.training.updater import TrainingUpdater, TestingUpdater
class GrammarTrainingUpdater(TrainingUpdater):
    @classmethod
    def from_bot(cls, bot: TrialBot) -> 'GrammarTrainingUpdater':
        updater = super().from_bot(bot)
        del updater._optims
        args, hparams, model = bot.args, bot.hparams, bot.model
        from radam import RAdam
        optim = RAdam(model.parameters(), lr=hparams.ADAM_LR, weight_decay=hparams.weight_decay)
        bot.logger.info("Use RAdam optimizer: " + str(optim))
        updater._optims = [optim]
        return updater

class GrammarTestingUpdater(TestingUpdater):
    def update_epoch(self):
        model, iterator, device = self._models[0], self._iterators[0], self._device
        model.eval()
        batch = next(iterator)
        if iterator.is_new_epoch:
            self.stop_epoch()
        if device >= 0:
            batch = move_to_device(batch, device)
        output = model(**batch)
        return output

def prediction_analysis(bot: TrialBot):
    # everything is (batch, derivation, rhs_seq - 1)
    # except for lhs is (batch, derivation)
    #
    # { "error": [is_nt_err, nt_err, t_err],
    #   "gold": [mask, out_is_nt, safe_nt_out, safe_t_out],
    #   "all_lhs": derivation_tree[:, :, 0], }
    #
    # preds = (is_nt_prob > 0.5, nt_logits.argmax(dim=-1), t_logits.argmax(dim=-1))
    #
    output = bot.state.output['deliberate_analysis']
    is_nt_err, nt_err, t_err = output['error']
    mask, out_is_nt, safe_nt_out, safe_t_out = output['gold']
    preds = bot.state.output['preds']
    all_lhs = output['all_lhs']
    print('===' * 30)
    batch_sz, drv_num = all_lhs.size()

    nt_tok = lambda k: bot.vocab.get_token_from_index(k, bot.hparams.ns[0])
    t_tok = lambda k: bot.vocab.get_token_from_index(k, bot.hparams.ns[1])

    def interweave_rule(is_nt, nt_toks, t_toks, padding_start):
        # all: (rhs_seq - 1,)
        expansion_id = [str(nt.item()) if cond > 0 else str(t.item())
                        for cond, nt, t in zip(is_nt, nt_toks, t_toks)]
        expansion_tok = [nt_tok(nt.item()) if cond > 0 else t_tok(t.item())
                         for cond, nt, t in zip(is_nt, nt_toks, t_toks)]
        expansion_id.insert(padding_start, '||')
        expansion_tok.insert(padding_start, '||')
        return expansion_id, expansion_tok

    for i in range(batch_sz):
        if mask[i][0][0].item() == 0:
            continue

        print('===' * 30)

        for j in range(drv_num):
            if mask[i][j][0].item() == 0:
                continue

            if j > 0:
                print('---' * 20)

            padding_start = mask[i][j].sum().item()

            gold_id, gold_tok = interweave_rule(out_is_nt[i][j], safe_nt_out[i][j], safe_t_out[i][j], padding_start)
            print(f'GOLD_ID:  {all_lhs[i][j].item()}   --> {" ".join(gold_id)}')
            print(f'GOLD_TOK: {nt_tok(all_lhs[i][j].item())} --> {" ".join(gold_tok)}')

            pred_medal = []
            for gz, z, nt, t in zip(out_is_nt[i][j], is_nt_err[i][j], nt_err[i][j], t_err[i][j]):
                z, nt, t = list(map(lambda n: 'x' if n > 0 else 'o', (z, nt, t)))
                medal = (z + nt) if gz > 0 else (z + t)
                pred_medal.append(medal)
            pred_medal.insert(padding_start, '|')

            pred_id, pred_tok = interweave_rule(preds[0][i][j], preds[1][i][j], preds[2][i][j], padding_start)
            print(f'PRED_ID:  {all_lhs[i][j].item()}   --> {" ".join(pred_id)}')
            print(f'PRED_TOK: {nt_tok(all_lhs[i][j].item())} --> {" ".join("%s(%s)" % t for t in zip(pred_tok, pred_medal))}')


def main():
    args = setup(seed=2021)
    from trialbot.training import Events
    bot = TrialBot(args, 'ebnf_lm', get_ebnf_lm)

    def get_metrics(bot: TrialBot):
        print(json.dumps(bot.model.get_metric(reset=True)))

    from utils.trial_bot_extensions import print_hyperparameters
    from trialbot.training.extensions import ext_write_info
    from trialbot.training.extensions import every_epoch_model_saver
    from utils.trial_bot_extensions import debug_models, end_with_nan_loss

    bot.add_event_handler(Events.STARTED, print_hyperparameters, 100)
    bot.add_event_handler(Events.STARTED, ext_write_info, 105, msg="-" * 50)
    bot.add_event_handler(Events.STARTED, debug_models, 100)
    bot.add_event_handler(Events.EPOCH_COMPLETED, get_metrics, 90)
    if not args.test:
        from utils.trial_bot_extensions import evaluation_on_dev_every_epoch
        bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 90)
        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)
        bot.add_event_handler(Events.ITERATION_COMPLETED, end_with_nan_loss, 100)
        bot.updater = GrammarTrainingUpdater.from_bot(bot)
    else:
        bot.updater = GrammarTestingUpdater.from_bot(bot)
        if args.debug:
            bot.add_event_handler(Events.ITERATION_COMPLETED, prediction_analysis, 100)
    bot.run()

if __name__ == '__main__':
    main()