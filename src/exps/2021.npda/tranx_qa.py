from typing import Dict, List, Tuple, Callable, Any
from os.path import join, abspath, dirname
import sys
sys.path.insert(0, abspath(join(dirname(__file__), '..', '..')))   # up to src
import torch
from trialbot.training import TrialBot, Events, Registry, Updater
from trialbot.data import RandomIterator
from trialbot.utils.move_to_device import move_to_device

import logging
import datasets.cfq
import datasets.cfq_translator

class TranXTrainingUpdater(Updater):
    def __init__(self, models, iterators, optims, device=-1,
                 dry_run: bool = False,
                 param_group_conf: List[Tuple[str, Callable[..., float]]] = None,
                 ):
        super().__init__(models, iterators, optims, device)
        self._dry_run = dry_run
        self.param_group_conf = param_group_conf

    def update_epoch(self):
        optim: torch.optim.Optimizer
        model, optim, iterator = self._models[0], self._optims[0], self._iterators[0]
        if iterator.is_new_epoch:
            self.stop_epoch()

        device = self._device
        model.train()
        optim.zero_grad()
        batch: Dict[str, torch.Tensor] = next(iterator)

        if device >= 0:
            batch = move_to_device(batch, device)

        output = model(**batch)
        if not self._dry_run:
            loss = output['loss']
            loss.backward()
            if self.param_group_conf is not None:
                for i, (_, factor_fn) in enumerate(self.param_group_conf):
                    optim.param_groups[i]['lr'] = optim.defaults['lr'] * factor_fn(batch)
            optim.step()
        return output

    @staticmethod
    def param_pattern_match(model: torch.nn.Module, group_conf: List[Tuple[str, Callable[..., float]]]):
        params = [{"params": []} for _ in range(len(group_conf) + 1)]
        logging.getLogger(__name__).info(f"param group len={len(params)}")
        for k, v in model.named_parameters():
            for i, (key_pref, _) in enumerate(group_conf):
                if k.startswith(key_pref):
                    params[i]["params"].append(v)
                    logging.getLogger(__name__).info(f"param {k} assigned to group {i}")
                    break
            else:
                params[-1]["params"].append(v)
                logging.getLogger(__name__).info(f"param {k} assigned to the last group")

        return params

    @classmethod
    def from_bot(cls, bot: TrialBot) -> 'TranXTrainingUpdater':
        args, p, model, logger = bot.args, bot.hparams, bot.model, bot.logger

        group_conf = getattr(p, 'group_conf', None)
        params = model.parameters() if group_conf is None else cls.param_pattern_match(model, group_conf)
        optim = cls.get_optim(p, params)
        logger.info(f"Using Optimizer {optim}")

        device, dry_run = args.device, args.dry_run
        repeat_iter = shuffle_iter = not args.debug
        iterator = RandomIterator(bot.train_set, bot.hparams.batch_sz, bot.translator,
                                  shuffle=shuffle_iter, repeat=repeat_iter)
        if args.debug and args.skip:
            iterator.reset(args.skip)

        updater = cls(model, iterator, optim, device, dry_run, group_conf)
        return updater

    @classmethod
    def get_optim(cls, p, params):
        if hasattr(p, "OPTIM") and isinstance(p.OPTIM, str) and p.OPTIM.lower() == "sgd":
            optim = torch.optim.SGD(params, p.SGD_LR, weight_decay=p.WEIGHT_DECAY)
        elif hasattr(p, "OPTIM") and isinstance(p.OPTIM, str) and p.OPTIM.lower() == "radam":
            from radam import RAdam
            optim = RAdam(params, lr=p.ADAM_LR, weight_decay=p.WEIGHT_DECAY)
        else:
            optim = torch.optim.Adam(params, p.ADAM_LR, p.ADAM_BETAS, weight_decay=p.WEIGHT_DECAY)
        return optim

def main():
    from utils.trialbot_setup import setup
    from models.base_s2s.base_seq2seq import BaseSeq2Seq
    args = setup(seed=2021, translator='cfq_tranx_mod_ent_qa', dataset='cfq_mcd1', hparamset='cfq_mod_ent_tranx')
    bot = TrialBot(trial_name='tranx_qa', get_model_func=BaseSeq2Seq.from_param_and_vocab, args=args)

    from trialbot.training import Events
    @bot.attach_extension(Events.EPOCH_COMPLETED)
    def get_metrics(bot: TrialBot):
        import json
        print(json.dumps(bot.model.get_metric(reset=True)))

    from utils.trial_bot_extensions import print_hyperparameters
    bot.add_event_handler(Events.STARTED, print_hyperparameters, 90)

    from trialbot.training import Events
    if not args.test:
        # --------------------- Training -------------------------------
        @bot.attach_extension(Events.STARTED)
        def print_models(bot: TrialBot):
            print(str(bot.models))

        from trialbot.training.extensions import every_epoch_model_saver
        from utils.trial_bot_extensions import end_with_nan_loss, save_model_every_num_iters
        from utils.trial_bot_extensions import evaluation_on_dev_every_epoch, collect_garbage
        bot.add_event_handler(Events.EPOCH_STARTED, collect_garbage, 95)
        bot.add_event_handler(Events.ITERATION_COMPLETED, end_with_nan_loss, 100)
        bot.add_event_handler(Events.ITERATION_COMPLETED, collect_garbage, 95)
        bot.add_event_handler(Events.ITERATION_COMPLETED, save_model_every_num_iters, 100)
        bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 90)
        bot.add_event_handler(Events.EPOCH_COMPLETED, collect_garbage, 95)
        # bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)

        # @bot.attach_extension(Events.ITERATION_COMPLETED)
        def batch_data_size(bot: TrialBot):
            output = bot.state.output
            if output is None:
                return

            print("source_size:", output['source'].size(),
                  "target_size:", output['target'].size(),
                  "pred_size:", output['predictions'].size())

        bot.updater = TranXTrainingUpdater.from_bot(bot)
    elif args.debug:
        @bot.attach_extension(Events.ITERATION_COMPLETED)
        def print_output(bot: TrialBot):
            import json
            output = bot.state.output
            if output is None:
                return

            model = bot.model
            output = model.revert_tensor_to_string(output)

            batch_print = []
            for src, gold, pred in zip(*map(output.get, ("source_tokens", "target_tokens", "predicted_tokens"))):
                to_print = f'SRC:  {" ".join(src)}\nGOLD: {" ".join(gold)}\nPRED: {" ".join(pred)}'
                batch_print.append(to_print)

            sep = '\n' + '-' * 60 + '\n'
            print(sep.join(batch_print))

    bot.run()

@Registry.hparamset()
def cfq_mod_ent_tranx():
    from trialbot.training.hparamset import HyperParamSet
    from trialbot.utils.root_finder import find_root
    p = HyperParamSet.common_settings(find_root())
    p.TRAINING_LIMIT = 200  # in num of epochs
    p.OPTIM = "RAdam"
    p.batch_sz = 32

    p.emb_sz = 256
    p.src_namespace = 'questionPatternModEntities'
    p.tgt_namespace = 'modent_rule_seq'
    p.hidden_sz = 256
    p.enc_attn = "bilinear"
    p.dec_hist_attn = "dot_product"
    p.concat_attn_to_dec_input = False
    p.encoder = "bilstm"
    p.num_enc_layers = 2
    p.dropout = .2
    p.decoder = "lstm"
    p.num_dec_layers = 2
    p.max_decoding_step = 100
    p.scheduled_sampling = .1
    p.decoder_init_strategy = "forward_last_parallel"
    p.tied_decoder_embedding = True
    return p

@Registry.hparamset()
def cfq_mod_ent_moderate_tranx():
    p = cfq_mod_ent_tranx()
    p.TRAINING_LIMIT = 5  # in num of epochs
    p.WEIGHT_DECAY = .1
    p.emb_sz = 128
    p.hidden_sz = 128
    p.dec_hist_attn = 'none'
    return p

@Registry.hparamset()
def cfq_mod_ent_moderate_tranx_scaled():
    p = cfq_mod_ent_tranx()
    p.TRAINING_LIMIT = 5  # in num of epochs
    p.WEIGHT_DECAY = .1
    p.emb_sz = 128
    p.hidden_sz = 128
    p.dec_hist_attn = 'none'

    src_len_fn = lambda batch: (batch['source_tokens'] != 0).sum(dim=-1).float().mean().item()
    tgt_len_fn = lambda batch: (batch['target_tokens'] != 0).sum(dim=-1).float().mean().item()

    group_conf: List[Tuple[str, Callable[..., float]]] = [
        # every target loss will be back-propagated
        ("_encoder", lambda batch: 1. / (src_len_fn(batch) * tgt_len_fn(batch))),
        ("_src_embedding", lambda batch: 1. / tgt_len_fn(batch)),

        # the i-th step receives (n - i) updates, yielding a total of n(n+1)/2
        ("_decoder", lambda batch: 2. / (tgt_len_fn(batch) * (1 + tgt_len_fn(batch)))),

        # encoder attention weights are updated by (N x M) times
        ("_enc_attn", lambda batch: 1. / (src_len_fn(batch) * tgt_len_fn(batch))),
        # decoder history attention weights are updated also by n(n+1)/2 times
        ("_dec_hist_attn", lambda batch: 2. / (tgt_len_fn(batch) * (1 + tgt_len_fn(batch)))),

        # target_embedding is tied to output projection, the parameters are updated by n + 1 times
        ("_tgt_embedding", lambda batch: 1. / (tgt_len_fn(batch) + 1)),
        # output projection is only bias weight when tied, which is updated by n times
        ("_output_projection", lambda batch: 1. / tgt_len_fn(batch)),
    ]
    p.group_conf = group_conf
    # only the batch scale is averaged out, other loss are summed together and rely on the group_conf
    p.training_average = 'bare_batch'
    return p

@Registry.hparamset()
def cfq_mod_ent_tiny_tranx():
    p = cfq_mod_ent_tranx()
    p.TRAINING_LIMIT = 5
    p.emb_sz = 32
    p.hidden_sz = 32
    p.encoder = 'lstm'
    p.dec_hist_attn = 'none'
    p.enc_attn = 'dot_product'
    return p

@Registry.hparamset()
def cfq_mod_ent_small_tranx():
    p = cfq_mod_ent_tiny_tranx()
    p.emb_sz = 128
    p.hidden_sz = 128
    return p

if __name__ == '__main__':
    main()
