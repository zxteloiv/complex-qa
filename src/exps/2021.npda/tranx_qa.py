from typing import Dict, List, Tuple, Callable, Any, Union
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
                 param_group_conf: List[Tuple[str, Callable[..., float], List[Union[None, int]]]] = None,
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
                group_grad_norms = []
                for group in optim.param_groups:
                    count = acc = 0
                    for p in group["params"]:
                        if p.grad is not None:
                            count += 1
                            acc += p.grad.abs().mean()
                    group_grad_norms.append(None if count == 0 else acc / count)

                for i, (_, factor_fn, dep_groups) in enumerate(self.param_group_conf):
                    # dep_grad_norms = []
                    # for g_id in dep_groups: # Union[None, int]
                    #     if g_id is None:
                    #         dep_grad_norms.append(1)
                    #         continue
                    #
                    #     group_grad_norm = group_grad_norms[g_id]
                    #     if group_grad_norm is not None:
                    #         # group grad norm is represented by the mean of all parameters in the group
                    #         dep_grad_norms.append(group_grad_norm)
                    #
                    # # delta by adding norms from different groups , and multiplied by the group num
                    # if len(dep_grad_norms) > 0:
                    #     delta = len(dep_grad_norms) / sum(dep_grad_norms)
                    # else:
                    #     delta = 1
                    optim.param_groups[i]['lr'] = optim.defaults['lr'] * factor_fn(batch) #delta # factor_fn(batch) * delta

            # lbfgs requires the closure but sgd
            # def closure():
            #     optim.zero_grad()
            #     loss = model(**batch)['loss']
            #     loss.backward()
            #     return loss

            optim.step()
        return output

    @staticmethod
    def param_pattern_match(model: torch.nn.Module, group_conf: List[Tuple[str, Callable[..., float], List[Union[None, int]]]]):
        params = [{"params": []} for _ in range(len(group_conf) + 1)]
        logging.getLogger(__name__).info(f"param group len={len(params)}")
        for k, v in model.named_parameters():
            for i, (key_pref, _, dep_groups) in enumerate(group_conf):
                if k.startswith(key_pref):
                    params[i]["params"].append(v)
                    logging.getLogger(__name__).info(f"param {k} assigned to group {i}, depends on group {dep_groups}")
                    break
            else:
                params[-1]["params"].append(v)
                logging.getLogger(__name__).info(f"param {k} assigned to the last group")

        return params

    @classmethod
    def from_bot(cls, bot: TrialBot) -> 'TranXTrainingUpdater':
        args, p, model, logger = bot.args, bot.hparams, bot.model, bot.logger
        from utils.select_optim import select_optim

        group_conf = getattr(p, 'group_conf', None)
        params = model.parameters() if group_conf is None else cls.param_pattern_match(model, group_conf)
        optim = select_optim(p, params)
        logger.info(f"Using Optimizer {optim}")

        device, dry_run = args.device, args.dry_run
        repeat_iter = shuffle_iter = not args.debug
        iterator = RandomIterator(bot.train_set, bot.hparams.batch_sz, bot.translator,
                                  shuffle=shuffle_iter, repeat=repeat_iter)
        if args.debug and args.skip:
            iterator.reset(args.skip)

        updater = cls(model, iterator, optim, device, dry_run, group_conf)
        return updater

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
        from utils.trial_bot_extensions import end_with_nan_loss
        from utils.trial_bot_extensions import evaluation_on_dev_every_epoch, collect_garbage
        bot.add_event_handler(Events.EPOCH_STARTED, collect_garbage, 95)
        bot.add_event_handler(Events.ITERATION_COMPLETED, end_with_nan_loss, 100)
        bot.add_event_handler(Events.ITERATION_COMPLETED, collect_garbage, 95)
        bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 90)
        bot.add_event_handler(Events.EPOCH_COMPLETED, collect_garbage, 95)
        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)

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
    p.TRAINING_LIMIT = 10  # in num of epochs
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
    p.TRAINING_LIMIT = 10  # in num of epochs
    p.WEIGHT_DECAY = .1
    p.emb_sz = 128
    p.hidden_sz = 128
    p.dec_hist_attn = "dot_product"
    return p

@Registry.hparamset()
def cfq_mod_ent_moderate_tranx_scaled():
    p = cfq_mod_ent_tranx()
    p.TRAINING_LIMIT = 10  # in num of epochs
    p.OPTIM = "adabelief"
    p.ADAM_LR = 1e-3
    p.WEIGHT_DECAY = .1
    p.ADAM_BETAS = (0.9, 0.999)
    p.optim_kwargs = {"eps": 1e-16}
    p.emb_sz = 128
    p.hidden_sz = 128
    p.dec_hist_attn = "dot_product"

    # src_len_fn = lambda batch: (batch['source_tokens'] != 0).sum(dim=-1).float().mean().item()
    # tgt_len_fn = lambda batch: (batch['target_tokens'] != 0).sum(dim=-1).float().mean().item()
    #
    # group_conf: List[Tuple[str, Callable[..., float], List[Union[None, int]]]] = [
    #     ("_decoder", lambda batch: src_len_fn(batch) * 1. / (1 + tgt_len_fn(batch)), [None]),
    # ]
    # group_conf: List[Tuple[str, Callable[..., float], List[Union[None, int]]]] = [
    #     ("_encoder", lambda batch: .5 / src_len_fn(batch), [2, 3]),   # 0
    #     ("_src_embedding", lambda batch: .5 / src_len_fn(batch), [0]),   # 1
    #
    #     ("_decoder", lambda batch: 1. / (1 + tgt_len_fn(batch)), [3, 6, 4]), # 2
    #
    #     ("_enc_attn", lambda batch: 1. / src_len_fn(batch), [6]),  # 3
    #     ("_dec_hist_attn", lambda batch: 1., [6]),   # 4
    #
    #     ("_tgt_embedding", lambda batch: 1., [None, 2]), # 5
    #     ("_output_projection", lambda batch: 1., [None]),   # 6
    # ]
    # p.group_conf = group_conf
    p.training_average = 'batch'
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
