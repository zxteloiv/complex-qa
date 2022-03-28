from trialbot.utils.grid_search_helper import import_grid_search_parameters
from trialbot.training import Registry, TrialBot
from trialbot.training.hparamset import HyperParamSet
from trialbot.utils.root_finder import find_root
import sys
sys.path.insert(0, find_root('.SRC'))


def main():
    from utils.trialbot.setup_cli import setup as setup_cli
    from libs2s import setup_common_bot
    import datasets.comp_gen_bundle as cg_bundle
    cg_bundle.install_parsed_qa_datasets(Registry._datasets)
    import datasets.cg_bundle_translator
    from models.rnng.model_factory import RNNGBuilder

    bot = setup_common_bot(
        args=setup_cli(translator='rnng', seed=2021, device=0),
        get_model_func=RNNGBuilder.from_param_and_vocab,
    )
    bot.run()


@Registry.hparamset('default')
def _base_hparams():
    p = HyperParamSet.common_settings(find_root())
    p.TRAINING_LIMIT = 150
    p.WEIGHT_DECAY = 0.
    p.OPTIM = "adabelief"
    p.ADAM_BETAS = (0.9, 0.999)
    p.batch_sz = 16

    # self.get_embeddings()
    p.emb_sz = 100
    p.src_namespace = "sent"
    p.src_emb_pretrained_file = "~/.glove/glove.6B.100d.txt.gz"
    p.tgt_namespace = "target"

    p.lr_scheduler_kwargs = {'model_size': 400, 'warmup_steps': 50}

    # self.get_embed_encoder_bundle()
    p.dropout = 0.2
    p.enc_dropout = 0.
    p.encoder = 'lstm'
    p.diora_loss_enabled = True

    # self.get_encoder_stack()
    p.diora_concat_outside = True
    p.hidden_sz = 200
    p.enc_out_dim = 200
    p.num_heads = 10
    p.use_cell_based_encoder = False
    p.cell_encoder_is_bidirectional = False
    p.cell_encoder_uses_packed_sequence = False
    p.num_enc_layers = 1

    # rnng
    p.rnng_namespaces = ('rnng', 'nonterminal', 'terminal')
    p.root_ns = 'grammar_entry'

    p.TRANSLATOR_KWARGS = dict(src_ns=p.src_namespace,
                               rnng_namespaces=p.rnng_namespaces,
                               grammar_entry_ns=p.root_ns,
                               )
    p.NS_VOCAB_KWARGS = dict(
        non_padded_namespaces=[p.root_ns, p.rnng_namespaces[1], p.rnng_namespaces[2]]
    )
    p.dec_dropout = p.dropout
    return p


if __name__ == '__main__':
    main()
