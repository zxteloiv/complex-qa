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

from utils.trialbot.extensions import print_hyperparameters, get_metrics, print_models
from utils.trialbot.extensions import save_multiple_models_per_epoch
from utils.trialbot.extensions import end_with_nan_loss
from utils.trialbot.extensions import collect_garbage

from utils.select_optim import select_optim
from trialbot.utils.move_to_device import move_to_device
from utils.trialbot.setup import setup
from utils.seq_collector import SeqCollector
from utils.lark.restore_cfg import restore_grammar_from_trees, export_grammar
from idioms.export_conf import get_export_conf
import lark


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
            if 'sql_tree' in raw:
                batch_data.append(raw)
        # ends of iteration will raise a StopIteration after the current batch, and then event engine will return None.
        # an empty batch is set to return None, but raising the StopIteration exception is not required.
        if iterator.is_end_of_epoch:
            self.stop_epoch()
        return batch_data

    def _run_policy_net(self, batch: list):
        tensor_list = [self.translator.to_tensor(x) for x in batch]

        filtered_tensors, filtered_batch = [], []
        for tensor, example in zip(tensor_list, batch):
            if all(v is not None for v in tensor.values()):
                filtered_tensors.append(tensor)
                filtered_batch.append(example)

        tensor_batch = self.translator.batch_tensor(filtered_tensors)
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
        for i, data in enumerate(filtered_batch):
            sample_iter = zip(nodes[i], actions[i], logprobs[i], probs[i])
            # if a sampled modification is failed, the example will de facto be omitted,
            # which may be changed to select some examples along with the original one (representing a staged grammar)
            # to train the parser, i.e., resembling the parser training side
            # the parser training doesn't require the existence of prob and logprob
            reset = True
            for node, action, prob, logprob in sample_iter:
                new_data = deepcopy(data)
                success = modify_tree(enrich_tree(new_data['runtime_tree']), node, action)
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


def collect_epoch_grammars(bot: TrialBot):
    bot.logger.info("Collecting grammar from the training trees...")
    train_trees = list(filter(None, [data['runtime_tree'] for data in bot.train_set]))
    if len(train_trees) == 0:
        raise ValueError("runtime tree of the entire training set not available.")

    g, terminals = restore_grammar_from_trees(train_trees)
    lex_file, start, export_terminals, excluded = get_export_conf(bot.args.dataset)
    lex_in = osp.join(find_root(), 'src', 'statics', 'grammar', lex_file)
    bot.logger.debug(f"Got grammar: start={start.name}, lex_in={lex_in}")
    g_txt = export_grammar(g, start, lex_in, terminals if export_terminals else None, excluded)
    bot.state.g_txt = g_txt
    grammar_filename = osp.join(bot.savepath, f"grammar_{bot.state.epoch}.lark")
    bot.logger.info(f"Grammar saved to {grammar_filename}")
    print(g_txt, file=open(grammar_filename, 'w'))
    parser = lark.Lark(bot.state.g_txt, start=start.name, keep_all_tokens=True)
    bot.state.parser = parser


def parse_and_eval_on_dev(bot: TrialBot, interval: int = 1,
                          clear_cache_each_batch: bool = True,
                          rewrite_eval_hparams: dict = None,
                          skip_first_epochs: int = 0,
                          on_test_data: bool = False,
                          ):
    from trialbot.utils.move_to_device import move_to_device
    import json
    if bot.state.epoch % interval == 0 and bot.state.epoch > skip_first_epochs:
        if on_test_data:
            bot.logger.info("Running for evaluation metrics on testing ...")
        else:
            bot.logger.info("Running for evaluation metrics ...")

        dataset, hparams = bot.dev_set, bot.hparams
        rewrite_eval_hparams = rewrite_eval_hparams or dict()
        for k, v in rewrite_eval_hparams.items():
            setattr(hparams, k, v)
        from trialbot.data import RandomIterator
        dataset = bot.test_set if on_test_data else bot.dev_set
        iterator = RandomIterator(len(dataset), hparams.batch_sz, shuffle=False, repeat=False)
        _, parser_model = bot.models
        device = bot.args.device
        parser_model.eval()
        lark_parser: lark.Lark = bot.state.parser
        for indices in iterator:
            raw_batch = []
            for index in indices:
                raw = dataset[index]
                try:
                    raw['runtime_tree'] = lark_parser.parse(raw['sql'])
                except:
                    bot.logger.warning(f'Failed to parse {dict((k, raw[k]) for k in ("sent", "sql"))}')
                    raw['runtime_tree'] = None
                raw_batch.append(raw)
            tensor_list = [bot.translator.to_tensor(example) for example in raw_batch]
            try:
                batch = bot.translator.batch_tensor(tensor_list)
            except:
                batch = None
            if batch is None or len(batch) == 0:
                continue
            if device >= 0:
                batch = move_to_device(batch, device)
            parser_model(batch['source_tokens'], batch['target_tokens'])

            if clear_cache_each_batch:
                import gc
                gc.collect()
                if bot.args.device >= 0:
                    import torch.cuda
                    torch.cuda.empty_cache()

        if on_test_data:
            get_metrics(bot, prefix="Testing Metrics: ")
        else:
            get_metrics(bot, prefix="Evaluation Metrics: ")


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

    bot.add_event_handler(Events.STARTED, print_hyperparameters, 90)
    bot.add_event_handler(Events.EPOCH_COMPLETED, get_metrics, 100)
    bot.add_event_handler(Events.STARTED, print_models, 100)

    @bot.attach_extension(Events.STARTED, 50)
    def print_snaptshot_path(bot: TrialBot):
        print("savepath:", bot.savepath)

    if not args.test:   # training behaviors
        bot.add_event_handler(Events.STARTED, collect_epoch_grammars, 80)
        bot.add_event_handler(Events.ITERATION_COMPLETED, end_with_nan_loss, 100)
        bot.add_event_handler(Events.ITERATION_COMPLETED, collect_garbage, 95)
        bot.add_event_handler(Events.EPOCH_COMPLETED, collect_epoch_grammars, 120)
        bot.add_event_handler(Events.EPOCH_COMPLETED, save_multiple_models_per_epoch, 100)
        bot.add_event_handler(Events.EPOCH_COMPLETED, parse_and_eval_on_dev, 90,
                              rewrite_eval_hparams={"batch_sz": 32})
        bot.add_event_handler(Events.EPOCH_COMPLETED, parse_and_eval_on_dev, 80,
                              rewrite_eval_hparams={"batch_sz": 32}, on_test_data=True)
        bot.updater = RS2GDTraining(bot)

    bot.run()


if __name__ == '__main__':
    main()
