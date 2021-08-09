# Reinforced Seq-to-Grammar-Derivation
import os.path as osp
import sys
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..', '..')))

from allennlp.modules.seq2seq_encoders import LstmSeq2SeqEncoder
from models.neural_pda.tree_action_policy import TreeActionPolicy
from models.base_s2s.stacked_encoder import StackedEncoder
from models.base_s2s.base_seq2seq import BaseSeq2Seq
from models.modules.attention_composer import get_attn_composer
from models.neural_pda import partial_tree_encoder as partial_tree
from models.modules.decomposed_bilinear import DecomposedBilinear

from trialbot.training import TrialBot, Events, Registry, Updater
from trialbot.data import NSVocabulary
import datasets.comp_gen_bundle as cg_bundle
cg_bundle.install_parsed_qa_datasets(Registry._datasets)
import datasets.cg_bundle_translator
from utils.select_optim import select_optim
from trialbot.utils.move_to_device import move_to_device
from utils.trialbot.setup import setup

class RS2GDTraining(Updater):
    def __init__(self, bot: TrialBot):
        args, p, model, logger = bot.args, bot.hparams, bot.model, bot.logger
        cluster_id = getattr(p, 'cluster_iter_key', None)
        if cluster_id is None:
            from trialbot.data.iterators import RandomIterator
            iterator = RandomIterator(len(bot.train_set), p.batch_sz)
            logger.info(f"Using RandomIterator with batch={p.batch_sz}")
        else:
            from trialbot.data.iterators import ClusterIterator
            iterator = ClusterIterator(bot.train_set, p.batch_sz, cluster_id)
            logger.info(f"Using ClusterIterator with batch={p.batch_sz} cluster_key={cluster_id}")

        params = model.parameters()
        optim = select_optim(p, params)
        self.dataset = bot.train_set
        self.translator = bot.translator
        super().__init__(model, iterator, optim, args.device)

    def update_epoch(self):
        batch_data = self._get_batch_data()
        if len(batch_data) == 0:
            return None

        logits, logprob, node_idx, action = self._run_policy_net(batch_data)

        return None

    def _get_batch_data(self):
        iterator = self._iterators[0]
        indices = next(iterator)
        batch_data = []
        for i in indices:
            raw = self.dataset[i]
            if 'sql_tree' not in raw:
                continue
            if 'runtime_tree' not in raw:
                raw['runtime_tree'] = raw['sql_tree']
            batch_data.append(raw)
        # ends of iteration will raise a StopIteration after the current batch, and then event engine will return None.
        # an empty batch is set to return None, but raising the StopIteration exception is not required.
        if iterator.is_end_of_epoch:
            self.stop_epoch()
        return batch_data

    def _run_policy_net(self, batch):
        tensor_list = [self.translator.to_tensor(x) for x in batch]
        batch = self.translator.batch_tensor(tensor_list)
        if self._device >= 0:
            batch = move_to_device(batch, self._device)

        model: TreeActionPolicy = self._models[0]
        optim = self._optims[0]
        optim.zero_grad()
        model.train()
        logits, logprob = model(batch['tree_nodes'], batch['node_pos'], batch['node_parents'])
        node_idx, action = model.decode(logprob)
        return logits, logprob, node_idx, action

@Registry.hparamset()
def crude_conf():
    from trialbot.utils.root_finder import find_root
    from trialbot.training.hparamset import HyperParamSet
    p = HyperParamSet.common_settings(find_root())

    p.batch_sz = 32

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
    p.action_num = 8
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
    args = setup(seed=2021)
    bot = TrialBot(trial_name='rs2gd', get_model_func=get_models, args=args)

    from utils.trialbot.extensions import print_hyperparameters, get_metrics, print_models
    bot.add_event_handler(Events.STARTED, print_hyperparameters, 90)
    bot.add_event_handler(Events.EPOCH_COMPLETED, get_metrics, 100)
    bot.add_event_handler(Events.EPOCH_COMPLETED, print_models, 100)

    if not args.test:   # training behaviors
        from trialbot.training.extensions import every_epoch_model_saver
        from utils.trialbot.extensions import end_with_nan_loss
        from utils.trialbot.extensions import collect_garbage
        bot.add_event_handler(Events.ITERATION_COMPLETED, end_with_nan_loss, 100)
        bot.add_event_handler(Events.ITERATION_COMPLETED, collect_garbage, 95)
        # from utils.trialbot.extensions import evaluation_on_dev_every_epoch
        # bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 90,
        #                       rewrite_eval_hparams={"batch_sz": 32})
        # bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 80,
        #                       rewrite_eval_hparams={"batch_sz": 32}, on_test_data=True)
        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)
        bot.updater = RS2GDTraining(bot)

    bot.run()

if __name__ == '__main__':
    main()