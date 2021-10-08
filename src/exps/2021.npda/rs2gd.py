# Reinforced Seq-to-Grammar-Derivation
import os.path as osp
import sys
import math
from trialbot.utils.root_finder import find_root
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..', '..')))

from allennlp.modules.seq2seq_encoders import LstmSeq2SeqEncoder
from models.neural_pda.tree_action_policy import TreeActionPolicy
from models.base_s2s.stacked_encoder import StackedEncoder
from models.base_s2s.base_seq2seq import BaseSeq2Seq
from models.modules.attention_composer import get_attn_composer
from models.neural_pda import partial_tree_encoder as partial_tree
from models.modules.decomposed_bilinear import DecomposedBilinear
from tree_worker import enrich_tree, modify_tree
import utils.nn as utilsnn
from copy import deepcopy

from trialbot.training import TrialBot, Events, Registry, Updater
from trialbot.data import NSVocabulary
import datasets.comp_gen_bundle as cg_bundle
cg_bundle.install_parsed_qa_datasets(Registry._datasets)
import datasets.cg_bundle_translator
from utils.select_optim import select_optim
from trialbot.utils.move_to_device import move_to_device
from utils.trialbot.setup import setup
from utils.seq_collector import SeqCollector


class RS2GDTraining(Updater):
    def __init__(self, bot: TrialBot):
        args, p, logger = bot.args, bot.hparams, bot.logger
        cluster_id = getattr(p, 'cluster_iter_key', None)
        from trialbot.data.iterators import RandomIterator
        iterator = RandomIterator(len(bot.train_set), p.batch_sz)
        logger.info(f"Using RandomIterator with batch={p.batch_sz}")

        models = list(bot.models)
        optims = [select_optim(p, m.parameters())for m in models]
        self.dataset = bot.train_set
        self.translator = bot.translator
        super().__init__(models, iterator, optims, args.device)

    def update_epoch(self):
        # batch_data: list of data (key-value pairs)
        batch_data = self._get_batch_data()
        if len(batch_data) == 0:
            return None

        augmented_data, prob, logprob = self._run_policy_net(batch_data)
        parser_output = self._run_parser(augmented_data)
        parser_loss, policy_loss = self._optim_step(parser_output, prob, logprob)

        output = {
            "loss": parser_loss + policy_loss,
            "parser_loss": parser_loss,
            "policy_loss": policy_loss,
        }

        return output

    def _get_batch_data(self):
        iterator = self._iterators[0]
        indices = next(iterator)
        batch_data = []
        for i in indices:
            raw = self.dataset[i]
            if 'sql_tree' not in raw:
                continue
            if 'runtime_tree' not in raw:
                raw['runtime_tree'] = enrich_tree(raw['sql_tree'])

            batch_data.append(raw)
        # ends of iteration will raise a StopIteration after the current batch, and then event engine will return None.
        # an empty batch is set to return None, but raising the StopIteration exception is not required.
        if iterator.is_end_of_epoch:
            self.stop_epoch()
        return batch_data

    def _run_policy_net(self, batch: list):
        tensor_list = [self.translator.to_tensor(x) for x in batch]
        tensor_batch = self.translator.batch_tensor(tensor_list)
        if self._device >= 0:
            tensor_batch = move_to_device(tensor_batch, self._device)

        model: TreeActionPolicy = self._models[0]
        model.train()
        output: dict = model(tensor_batch['tree_nodes'], tensor_batch['node_pos'], tensor_batch['node_parents'])
        out_logprob, out_prob = model.get_logprob(output['logits'], output['node_mask'])

        # sample some policies (size S) for each tree example in the batch
        # list, list, (B, S), (B, S)
        nodes, actions, logprobs, probs = model.decode(out_prob, out_logprob, method="topk")

        mem = SeqCollector()
        for i, data in enumerate(batch):
            sample_iter = zip(nodes[i], actions[i], logprobs[i], probs[i])
            # if a sampled modification is failed, the example will de facto be omitted,
            # which may be changed to select some examples along with the original one (representing a staged grammar)
            # to train the parser, i.e., resembling the parser training side
            # the parser training doesn't require the existence of prob and logprob
            reset = True
            for node, action, prob, logprob in sample_iter:
                new_data = deepcopy(data)
                success = modify_tree(new_data['runtime_tree'], node, action)
                if success:
                    mem(data=new_data, prob=prob, logprob=logprob)
                    if reset:
                        # only the first successful modification sampled will be saved into the future
                        reset = False
                        self.dataset[new_data['id']] = new_data

        # list, (B * S,), (B * S,)
        return mem['data'], mem.get_stacked_tensor('prob', dim=0), mem.get_stacked_tensor('logprob', dim=0)

    def _run_parser(self, batch):
        tensor_list = [self.translator.to_tensor(x) for x in batch]
        tensor_batch = self.translator.batch_tensor(tensor_list)
        if self._device >= 0:
            tensor_batch = move_to_device(tensor_batch, self._device)
        model: BaseSeq2Seq = self._models[1]
        model.train()
        output = model(tensor_batch['source_tokens'], tensor_batch['target_tokens'])
        return output

    def _optim_step(self, parser_output: dict, prob, logprob):
        """
        :param parser_output: dict
        :param prob: (B * S,)
        :param logprob: (B * S,)
        """
        for m, opt in zip(self._models, self._optims):
            m.train()
            opt.zero_grad()

        parser_loss = parser_output['loss']
        parser_loss.backward()

        # (B * S, len - 1, num_classes)
        logits = parser_output['logits']
        # (B * S, len - 1)
        target = parser_output['target'][:, 1:].contiguous()
        target_mask = (target != 0).long()
        # (B * S,)
        reward = utilsnn.masked_reducing_gather(utilsnn.logits_to_prob(logits, "bounded"), target, target_mask, 'none')

        bounded_policy_reward = (reward + reward / (prob + 1e-20)).detach()

        policy_loss = - (logprob * bounded_policy_reward).mean() + math.log(2)
        policy_loss.backward()

        for opt in self._optims:
            opt.step()

        return parser_loss, policy_loss


def eval_on_epoch_grammars(bot: TrialBot):
    for data in bot.train_set:
        pass

    pass


@Registry.hparamset()
def crude_conf():
    from trialbot.training.hparamset import HyperParamSet
    p = HyperParamSet.common_settings(find_root())

    p.batch_sz = 16

    # policy net params
    p.ns_symbols = 'ns_lf'
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
    p.action_num = 7
    p.max_children_num = 12

    p.TRANSLATOR_KWARGS = {"max_node_position": p.max_children_num}

    # parser params
    p.emb_sz = 256
    p.src_namespace = 'sent'
    p.tgt_namespace = 'rule_seq'
    p.hidden_sz = 128
    p.enc_out_dim = p.hidden_sz # by default
    p.dec_in_dim = p.hidden_sz  # by default
    p.dec_out_dim = p.hidden_sz  # by default

    p.tied_decoder_embedding = True
    p.proj_in_dim = p.emb_sz  # by default

    p.enc_attn = "bilinear"
    p.dec_hist_attn = "dot_product"
    p.dec_inp_composer = 'cat_mapping'
    p.dec_inp_comp_activation = 'mish'
    p.proj_inp_composer = 'cat_mapping'
    p.proj_inp_comp_activation = 'mish'
    p.encoder = "bilstm"
    p.num_enc_layers = 2
    p.decoder = "lstm"
    p.num_dec_layers = 2
    p.dropout = .2
    p.enc_dropout = p.dropout  # by default
    p.dec_dropout = p.dropout  # by default
    p.max_decoding_step = 100
    p.scheduled_sampling = .1
    p.decoder_init_strategy = "forward_last_parallel"
    # p.src_emb_pretrained_file = "~/.glove/glove.6B.100d.txt.gz"

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
        node_action_mapper=nn.Linear(p_hid_dim, p.action_num)
    )

    parser_net = BaseSeq2Seq.from_param_and_vocab(p, vocab)

    return policy_net, parser_net

def main():
    args = setup(seed=2021, hparamset='crude_conf', translator="gd")
    bot = TrialBot(trial_name='rs2gd', get_model_func=get_models, args=args)

    from utils.trialbot.extensions import print_hyperparameters, get_metrics, print_models
    bot.add_event_handler(Events.STARTED, print_hyperparameters, 90)
    # bot.add_event_handler(Events.EPOCH_COMPLETED, get_metrics, 100)
    bot.add_event_handler(Events.STARTED, print_models, 100)

    if not args.test:   # training behaviors
        from utils.trialbot.extensions import save_multiple_models_per_epoch
        from utils.trialbot.extensions import end_with_nan_loss
        from utils.trialbot.extensions import collect_garbage
        bot.add_event_handler(Events.ITERATION_COMPLETED, end_with_nan_loss, 100)
        bot.add_event_handler(Events.ITERATION_COMPLETED, collect_garbage, 95)
        from utils.trialbot.extensions import evaluation_on_dev_every_epoch
        # bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 90,
        #                       rewrite_eval_hparams={"batch_sz": 32})
        # bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 80,
        #                       rewrite_eval_hparams={"batch_sz": 32}, on_test_data=True)
        bot.add_event_handler(Events.EPOCH_COMPLETED, save_multiple_models_per_epoch, 100)
        bot.updater = RS2GDTraining(bot)

    bot.run()

if __name__ == '__main__':
    main()