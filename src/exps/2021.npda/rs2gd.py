# Reinforced Seq-to-Grammar-Derivation
from typing import Optional
import os, os.path as osp
import logging
import sys
import math
import random
from copy import deepcopy
from datetime import datetime as dt
import torch
import torch.nn.functional
import lark
from trialbot.utils.root_finder import find_root
from trialbot.data.iterators import RandomIterator
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..', '..')))

from allennlp.modules.seq2seq_encoders import LstmSeq2SeqEncoder
from models.neural_pda.tree_action_policy import TreeActionPolicy
from models.base_s2s.stacked_encoder import StackedEncoder
from models.modules.attention_composer import get_attn_composer
from models.neural_pda.seq2pda import Seq2PDA
from models.neural_pda import partial_tree_encoder as partial_tree
from models.modules.decomposed_bilinear import DecomposedBilinear
from utils.tree_mod import modify_tree
from utils.trialbot.reset import reset
from utils.lark.id_tree import build_from_lark_tree

from trialbot.training import TrialBot, Events, Registry, Updater
from trialbot.data import NSVocabulary
import datasets.comp_gen_bundle as cg_bundle
cg_bundle.install_parsed_qa_datasets(Registry._datasets)

from utils.trialbot.extensions import print_hyperparameters, get_metrics, print_models, print_snaptshot_path
from utils.trialbot.extensions import save_multiple_models_per_epoch
from utils.trialbot.extensions import end_with_nan_loss
from utils.trialbot.extensions import collect_garbage

from utils.select_optim import select_optim
from trialbot.utils.move_to_device import move_to_device
from utils.trialbot.setup import setup, setup_null_argv
from utils.lark.restore_cfg import export_grammar
from utils.cfg import restore_grammar_from_trees
from idioms.export_conf import get_export_conf
import s2pda_qa


class PolicyTraining(Updater):
    def __init__(self, bot: TrialBot):
        args, p, logger = bot.args, bot.hparams, bot.logger
        iterator = RandomIterator(len(bot.train_set), p.batch_sz)
        logger.info(f"Using RandomIterator with batch={p.batch_sz}")

        super().__init__(bot.models, iterator, select_optim(p, bot.model.parameters()), args.device)

        self.dataset = bot.train_set
        self.translator = bot.translator
        self.logger = logging.getLogger(self.__class__.__name__)
        self.envbot: Optional[TrialBot] = None

        self.update_dataset: bool = False

        self.accept_modification_ratio = p.accept_modification_ratio
        self.decay_rate = p.decay_rate

    def update_epoch(self):
        # batch_data: list of data (key-value pairs)
        batch_data = self._get_batch_data()
        if len(batch_data) == 0:
            return None

        tensor_batch, filtered_batch = self._get_tensor_batch(batch_data)
        # logprob, prob: (B, N, A)
        out_logprob, out_prob = self._get_policy_logits(tensor_batch)

        # list[B*S], (B, S), (B, S)
        sampled_data, sampled_prob, sampled_logp = self._sample_modifications(out_logprob, out_prob, filtered_batch)
        batch_sz, sample_num = sampled_prob.size()
        # reward: (B, S)
        reward = self._get_env_reward(sampled_data, batch_sz, sample_num)
        policy_loss = self._optim_step(reward, sampled_prob, sampled_logp)
        if self.update_dataset:     # a flag set to False during warming-up
            self._update_training_set(sampled_data, sample_num, reward)
        output = {"loss": policy_loss}
        return output

    def _get_batch_data(self):
        iterator = self._iterators[0]
        indices = next(iterator)
        batch_data = []
        for i in indices:
            raw = self.dataset[i]
            if raw.get('sql_tree') is not None:
                batch_data.append(raw)

        if iterator.is_end_of_epoch:
            self.stop_epoch()
        return batch_data

    def _get_tensor_batch(self, batch):
        tensor_list = [self.translator.to_tensor(x) for x in batch]
        filtered_tensors, filtered_batch = [], []
        # filter will be applied to tensors, but the raw data batch is required to build extension
        for tensor, example in zip(tensor_list, batch):
            if all(v is not None for v in tensor.values()):
                filtered_tensors.append(tensor)
                filtered_batch.append(example)
        tensor_batch = self.translator.batch_tensor(filtered_tensors)

        if self._device >= 0:
            tensor_batch = move_to_device(tensor_batch, self._device)

        return tensor_batch, filtered_batch

    def _get_policy_logits(self, tensor_batch):
        model: TreeActionPolicy = self._models[0]
        model.train()
        output: dict = model(tensor_batch['tree_nodes'], tensor_batch['node_pos'], tensor_batch['node_parents'])
        # logprob, prob: (B, N, A)
        out_logprob, out_prob = model.get_logprob(output['logits'], output['node_mask'], tensor_batch['action_mask'])
        return out_logprob, out_prob

    def _sample_modifications(self, out_logprob, out_prob, batch):
        # out_logprob, out_prob: (B, N, A)
        model: TreeActionPolicy = self._models[0]
        # sample some policies (size S) for each tree example in the batch
        # (B, S), (B, S), pure tensors (no backwards)
        nodes, actions = model.decode(out_prob, out_logprob, method="topk", sample_num=5)

        sampled_data = []
        for i, data in enumerate(batch):
            for sample_j, (node_id, action_id) in enumerate(zip(nodes[i], actions[i])):
                new_data = deepcopy(data)
                modify_tree(new_data['runtime_tree'], node_id.item(), action_id.item())
                sampled_data.append(new_data)

        # (B, 1)
        batch_dim_indices = torch.arange(len(batch), device=out_prob.device).unsqueeze(-1)
        # (B, S)
        sampled_prob = out_prob[batch_dim_indices, nodes, actions]
        sampled_logp = out_logprob[batch_dim_indices, nodes, actions]

        # list[B*S], (B, S), (B, S)
        return sampled_data, sampled_prob, sampled_logp

    def _get_env_reward(self, sampled_batch, batch_sz: int, sample_num: int) -> torch.Tensor:
        translator = self.envbot.translator
        tensor_list = [translator.to_tensor(x) for x in sampled_batch]
        tensor_batch = translator.batch_tensor(tensor_list)
        if self._device >= 0:
            tensor_batch = move_to_device(tensor_batch, self._device)

        model: Seq2PDA = self.envbot.model
        model.eval()
        with torch.no_grad():
            output = model(model_function="validation",
                           source_tokens=tensor_batch['source_tokens'],
                           tree_nodes=tensor_batch['tree_nodes'],
                           node_parents=tensor_batch['node_parents'],
                           expansion_frontiers=tensor_batch['expansion_frontiers'],
                           derivations=tensor_batch['derivations'],
                           )
        # reward: (B, S)
        env_reward = output['reward']
        logit_reward = env_reward.reshape(batch_sz, sample_num)
        # add a high temperature to the reward computation
        # reward = torch.nn.functional.softmax(logit_reward * sample_num, dim=-1).detach_()
        # reward = torch.sigmoid(logit_reward).detach_()
        # std, mean = torch.std_mean(logit_reward, dim=-1, keepdim=True, unbiased=False)
        # reward = ((logit_reward - mean) / std).sigmoid()
        reward = logit_reward * (logit_reward > 0)   # the rule scorer must be greater than 0
        return reward

    def _update_training_set(self, sampled_batch, sample_num: int, reward):
        # updating the training example to the new sample with the largest reward
        # max_idx: (B,)
        _, max_idx = torch.max(reward.detach(), dim=-1)
        for i, data in enumerate(sampled_batch):
            row = i // sample_num
            col = i % sample_num
            if col == max_idx[row].item() and random.random() < self.accept_modification_ratio:
                self.dataset[data['id']] = data

    def _optim_step(self, reward, sampled_prob, sampled_logp):
        """
        :param reward: (B, S), the env_reward
        :param sampled_prob: (B, S)
        :param sampled_logp: (B, S)
        """
        for m, opt in zip(self._models, self._optims):
            m.train()
            opt.zero_grad()

        bounded_policy_reward = (reward + reward / (sampled_prob + 1e-20)).detach()
        policy_loss = - (sampled_logp * bounded_policy_reward).mean()
        self.logger.info(f"step reward: {reward.mean().item():.8f}")

        policy_loss.backward()
        for opt in self._optims:
            opt.step()

        return policy_loss


def accept_modification_schedule(bot: TrialBot):
    epoch = bot.state.epoch
    p = bot.hparams
    updater: PolicyTraining = bot.updater

    if epoch <= p.policy_warmup_epoch:
        logging.info(f"Using the reject_ratio {updater.accept_modification_ratio:6.4f} for tree modifications")
        return

    if epoch == p.policy_warmup_epoch + 1:
        logging.info(f"the training set will get updated along the policy net training from now on")
        updater.update_dataset = True

    if epoch > p.policy_warmup_epoch + 1:
        # the reject ratio will keep decreasing during the training process,
        # thus in a later epoch, the modifications will be less likely to get accepted.
        updater.accept_modification_ratio *= updater.decay_rate
        logging.info(f"Rejection Ratio decayed to {updater.accept_modification_ratio:6.4f}")


def cold_start(bot: TrialBot):
    args, p, logger = bot.args, bot.hparams, bot.logger
    envpath = osp.join(bot.savepath, f'env-0-cold_start')
    os.makedirs(envpath, exist_ok=True)

    logger.info(f'Running cold start for env model within {envpath}... {dt.now().strftime("%H:%M:%S")}')

    backup_handlers = logging.root.handlers[:]
    assert all(hasattr(p, a) for a in ('nested_translator', 'nested_hparamset'))

    nested_args = {"translator": p.nested_translator,
                   "dataset": args.dataset,
                   "snapshot-dir": envpath,
                   "hparamset": p.nested_hparamset,
                   "vocab-dump": osp.join(bot.savepath, 'vocab'),
                   "device": 0}
    nested_bot = s2pda_qa.make_trialbot(setup_null_argv(**nested_args))

    # binding the same datasets
    nested_bot.datasets = bot.datasets
    nested_bot.updater.dataset = bot.train_set

    pretrained_env_path = getattr(p, 'pretrained_env_model', None)
    if pretrained_env_path is None:
        # nested_logfile = osp.join(envpath, f'cold-start-training.log')
        # logger.info(f'Writing following logs into {nested_logfile} ')
        # logging.basicConfig(filename=nested_logfile, force=True)
        # logging.basicConfig(handlers=backup_handlers, force=True)
        nested_bot.run(p.cold_start_epoch)
    else:
        logging.info(f'loading the pretrained env model from {pretrained_env_path} ...')
        nested_bot.model.load_state_dict(torch.load(pretrained_env_path))
        if args.device >= 0:
            nested_bot.model.cuda(args.device)

    nested_bot.hparams.TRAINING_LIMIT = p.finetune_epoch
    bot.updater.envbot = nested_bot


def collect_epoch_grammars(bot: TrialBot, update_train_set: bool = False, update_runtime_parser: bool = True):
    p = bot.hparams
    if 0 < bot.state.epoch <= p.policy_warmup_epoch:
        return

    bot.logger.info("Collecting grammar from the training trees...")
    train_trees = list(filter(None, [data['runtime_tree'] for data in bot.train_set]))
    if len(train_trees) == 0:
        raise ValueError("runtime tree of the entire training set not available.")

    lex_file, start, export_terminals, excluded = get_export_conf(bot.args.dataset)
    lex_in = osp.join(find_root(), 'src', 'statics', 'grammar', lex_file)
    bot.logger.debug(f"Got grammar: start={start.name}, lex_in={lex_in}")

    # by default, the trees are converted from lark.Trees without introducing the terminal category nodes.
    # so when restoring the grammars, it is not needed to export the terminals, which means:
    #       export_terminal_values = False, and treat_terminals_as_categories = False,
    # the terminals will be None and thus will not affect the export_grammar function
    g, terminals = restore_grammar_from_trees(train_trees, export_terminal_values=False)
    g_txt = export_grammar(g, start, lex_in, terminals if export_terminals else None, excluded,
                           treat_terminals_as_categories=False,)
    grammar_filename = osp.join(bot.savepath, f"grammar_{bot.state.epoch}.lark")
    if update_train_set:
        grammar_filename = osp.join(bot.savepath, f"grammar_{bot.state.epoch}.upd.lark")
    bot.logger.info(f"Grammar saved to {grammar_filename}")
    print(g_txt, file=open(grammar_filename, 'w'))

    if update_runtime_parser:
        bot.state.g_txt = g_txt
        parser = lark.Lark(bot.state.g_txt, start=start.name, keep_all_tokens=True)
        bot.state.parser = parser

        if update_train_set:
            logging.info(f'Updating the train dataset by new grammar ...')
            updating_trees(parser, bot.train_set)


def updating_trees(parser, dataset):
    for i, x in enumerate(dataset):
        try:
            t = parser.parse(x['sql'])
            x['sql_tree'] = t
            x['runtime_tree'] = build_from_lark_tree(t, add_eps_nodes=True)
        except KeyboardInterrupt:
            raise SystemExit("Received Keyboard Interrupt and Exit now.")
        except:
            logging.warning(f'failed to parse the example {i}: {x["sent"]}')
            x['sql_tree'] = None
            x['runtime_tree'] = None

        dataset[x['id']] = x


def updating_dev_and_test_dataset(bot: TrialBot):
    parser = bot.state.parser
    p = bot.hparams
    if bot.state.epoch <= p.policy_warmup_epoch:
        return
    logging.info(f'Updating the dev dataset by new grammar ...')
    updating_trees(parser, bot.dev_set)
    logging.info(f'Updating the test dataset by new grammar ...')
    updating_trees(parser, bot.test_set)


def finetune_env_model(bot: TrialBot):
    updater: PolicyTraining = bot.updater
    epoch = bot.state.epoch
    if bot.state.epoch <= bot.hparams.policy_warmup_epoch:
        return

    logging.info('Start model fine-tuning on the updated training set')

    reset(updater.envbot)
    nested_path = osp.join(bot.savepath, f'nested-env-ep-{epoch}')
    updater.envbot.savepath = nested_path
    backup_handlers = logging.root.handlers[:]
    nested_logfile = osp.join(nested_path, f'epoch-{epoch}-finetuning.log')
    # logging.basicConfig(filename=nested_logfile, force=True)
    updater.envbot.run()
    # logging.basicConfig(handlers=backup_handlers, force=True)


@Registry.hparamset()
def crude_conf():
    from trialbot.training.hparamset import HyperParamSet
    p = HyperParamSet.common_settings(find_root())

    p.batch_sz = 16
    p.TRAINING_LIMIT = 60

    # policy net params
    p.node_emb_sz = 100
    p.pos_emb_sz = 50
    p.pos_enc_out = 100
    p.feature_composer = "cat_mapping"
    p.policy_hid_sz = 200
    p.policy_feature_activation = "tanh"
    p.num_re0_layer = 6
    p.bilinear_rank = 1
    p.bilinear_pool = 4
    p.bilinear_linear = True
    p.bilinear_bias = True
    p.action_num = 6
    p.max_children_num = 12
    p.accept_modification_ratio = .6
    p.decay_rate = .8

    # parser params
    p.nested_translator = 'cg_sql_pda'
    p.nested_hparamset = 'sql_pda'
    p.cold_start_epoch = 30
    p.finetune_epoch = 20
    p.src_namespace = 'sent'
    p.tgt_namespace = 'symbol'

    p.policy_warmup_epoch = 30
    p.pretrained_env_model = None
    return p


def get_models(p, vocab: NSVocabulary):
    from torch import nn
    p_hid_dim = p.policy_hid_sz
    policy_net = TreeActionPolicy(
        node_embedding=nn.Embedding(vocab.get_vocab_size(p.tgt_namespace), p.node_emb_sz),
        pos_embedding=nn.Embedding(p.max_children_num, p.pos_emb_sz),
        topo_pos_encoder=StackedEncoder([LstmSeq2SeqEncoder(p.pos_emb_sz, p.pos_enc_out)], p.pos_emb_sz, p.pos_enc_out),
        feature_composer=get_attn_composer(
            p.feature_composer, p.pos_enc_out, p.node_emb_sz, p_hid_dim,
            activation=p.policy_feature_activation
        ),
        tree_encoder=partial_tree.ReZeroEncoder(
            num_layers=p.num_re0_layer,
            layer_encoder=partial_tree.SingleStepBilinear(DecomposedBilinear(
                p_hid_dim, p_hid_dim, p_hid_dim, p.bilinear_rank, p.bilinear_pool,
                use_linear=p.bilinear_linear, use_bias=p.bilinear_bias,
            )),
        ),
        node_action_mapper=nn.Linear(p_hid_dim, p.action_num),
    )
    return policy_net


def main():
    args = setup(seed=2021, hparamset='crude_conf', translator="cg_sql_pda", device=0, dataset='scholar_iid.handcrafted')
    bot = TrialBot(trial_name='rgs_policy', get_model_func=get_models, args=args)

    bot.add_event_handler(Events.STARTED, print_hyperparameters, 90)
    bot.add_event_handler(Events.STARTED, print_models, 100)
    bot.add_event_handler(Events.STARTED, print_snaptshot_path, 50)

    if not args.test:   # training behaviors
        bot.add_event_handler(Events.STARTED, collect_epoch_grammars, 80)
        bot.add_event_handler(Events.STARTED, cold_start, 80)

        bot.add_event_handler(Events.ITERATION_COMPLETED, end_with_nan_loss, 100)
        bot.add_event_handler(Events.ITERATION_COMPLETED, collect_garbage, 95)
        bot.add_event_handler(Events.EPOCH_COMPLETED, save_multiple_models_per_epoch, 100)

        bot.add_event_handler(Events.EPOCH_STARTED, accept_modification_schedule, 100)
        bot.add_event_handler(Events.EPOCH_COMPLETED, collect_epoch_grammars, 100, update_train_set=True)
        bot.add_event_handler(Events.EPOCH_COMPLETED, collect_epoch_grammars, 100, update_runtime_parser=False)
        bot.add_event_handler(Events.EPOCH_COMPLETED, updating_dev_and_test_dataset, 100)
        bot.add_event_handler(Events.EPOCH_COMPLETED, finetune_env_model, 100)
        bot.updater = PolicyTraining(bot)

    bot.run()


if __name__ == '__main__':
    main()
