from trialbot.training import Registry
from trialbot.training.hparamset import HyperParamSet
from trialbot.utils.root_finder import find_root
import sys
sys.path.insert(0, find_root('.SRC'))


def main():
    from utils.trialbot.setup_cli import setup as setup_cli
    from utils.s2s_arch.setup_bot import setup_common_bot
    import shujuji.comp_gen_bundle as cg_bundle
    cg_bundle.install_parsed_qa_datasets(Registry._datasets)
    from models.pcfg.seq2pcfg_factory import Seq2PCFGBuilder

    bot = setup_common_bot(
        args=setup_cli(translator='seq2pcfg', seed=2021, device=0),
        get_model_func=Seq2PCFGBuilder.from_param_and_vocab,
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
    p.tgt_namespace = "sql"
    p.lr_scheduler_kwargs = {'model_size': 400, 'warmup_steps': 50}

    # self.get_embed_encoder_bundle()
    p.dropout = 0.2
    p.enc_dropout = 0.
    p.encoder = 'bilstm'
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

    # pcfg
    p.decoder = 'tdpcfg'
    p.num_pcfg_nt = 100
    p.num_pcfg_pt = 200
    p.td_pcfg_rank = p.num_pcfg_nt // 10
    p.pcfg_hidden_dim = p.hidden_sz
    return p


if __name__ == '__main__':
    main()
