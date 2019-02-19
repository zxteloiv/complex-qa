import os.path
import datetime
import torch
import torch.nn
import tqdm

import allennlp.data
from allennlp.data.iterators import BucketIterator

import allennlp.training

import allennlp
import allennlp.common
import allennlp.models
import allennlp.modules
import allennlp.predictors
from allennlp.modules.attention import BilinearAttention, DotProductAttention

import config

import data_adapter
import utils.opt_parser
from models.adaptive_seq2seq import AdaptiveSeq2Seq
from models.transformer.multi_head_attention import SingleTokenMHAttentionWrapper, GeneralMultiHeadAttention
from utils.nn import AllenNLPAttentionWrapper
from models.transformer.encoder import TransformerEncoder
from models.adaptive_rnn_cell import AdaptiveRNNCell, AdaptiveStateMode
from allennlp.common.util import START_SYMBOL, END_SYMBOL


def main():
    parser = utils.opt_parser.get_trainer_opt_parser()
    parser.add_argument('models', nargs='*', help='pretrained models for the same setting')
    parser.add_argument('--test', action="store_true", help='use testing mode')
    parser.add_argument('--emb-dim', type=int, help='basic embedding dimension')
    parser.add_argument('--num-layer', type=int, help='maximum number of stacked layers')
    parser.add_argument('--use-act', action="store_true", help='Use adaptive computation time for decoder')
    parser.add_argument('--act-loss-weight', type=float, help="the loss of the act weights")

    parser.add_argument('--act-mode', choices=['basic', 'random', 'mean_field'])
    parser.add_argument('--depth-emb', choices=['sinusoid', 'learnt', 'none'])

    # parser.add_argument('--decoder-attention', choices=["dot_product", "bilinear", "multihead"],
    #                    help="the attention used in decoder, dot_product might be best")

    args = parser.parse_args()

    reader = data_adapter.GeoQueryDatasetReader()
    training_set = reader.read(config.DATASETS[args.dataset].train_path)
    try:
        validation_set = reader.read(config.DATASETS[args.dataset].dev_path)
    except:
        validation_set = None

    vocab = allennlp.data.Vocabulary.from_instances(training_set)
    st_ds_conf = config.ADA_TRANS2SEQ_CONF[args.dataset]
    if args.num_layer:
        st_ds_conf['max_num_layers'] = args.num_layer
    if args.epoch:
        config.TRAINING_LIMIT = args.epoch
    if args.batch:
        st_ds_conf['batch_sz'] = args.batch
    if args.use_act:
        st_ds_conf['act'] = True
    # if args.decoder_attention:
    #     st_ds_conf['decoder_attn'] = args.decoder_attention
    if args.emb_dim:
        st_ds_conf['emb_sz'] = args.emb_dim
    if args.act_loss_weight:
        st_ds_conf['act_loss_weight'] = args.act_loss_weight
    if args.act_mode:
        st_ds_conf['act_mode'] = args.act_mode
    if args.depth_emb:
        st_ds_conf['depth_emb'] = args.depth_emb

    bsz = st_ds_conf['batch_sz']
    emb_sz = st_ds_conf['emb_sz']

    source_embedding = allennlp.modules.Embedding(num_embeddings=vocab.get_vocab_size('nltokens'),
                                                  embedding_dim=emb_sz)
    target_embedding = allennlp.modules.Embedding(num_embeddings=vocab.get_vocab_size('lftokens'),
                                                  embedding_dim=emb_sz)
    encoder = TransformerEncoder(input_dim=emb_sz,
                                 num_layers=st_ds_conf['num_enc_layers'],
                                 num_heads=st_ds_conf['num_heads'],
                                 feedforward_hidden_dim=emb_sz,
                                 attention_dropout=st_ds_conf['attention_dropout'],
                                 residual_dropout=st_ds_conf['residual_dropout'],
                                 feedforward_dropout=st_ds_conf['feedforward_dropout'],
                                 )

    def _get_attention():
        decoder_attn = "dot_product" #st_ds_conf['decoder_attn']
        if decoder_attn == "bilinear":
            attn = BilinearAttention(vector_dim=emb_sz, matrix_dim=emb_sz)
            attn = AllenNLPAttentionWrapper(attn)
        elif decoder_attn == "dot_product":
            attn = DotProductAttention()
            attn = AllenNLPAttentionWrapper(attn)
        elif decoder_attn == "multihead":
            attn = GeneralMultiHeadAttention(num_heads=st_ds_conf['num_heads'],
                                          input_dim=emb_sz,
                                          total_attention_dim=emb_sz,
                                          total_value_dim=emb_sz,
                                          attention_dropout=st_ds_conf['attention_dropout'],
                                          use_future_blinding=False,
                                          )
            attn = SingleTokenMHAttentionWrapper(attn)
        else:
            assert False
        return attn

    rnn_cell = torch.nn.LSTMCell(emb_sz, emb_sz)
    dwa = _get_attention() if st_ds_conf['dwa'] else None
    decoder = AdaptiveRNNCell(hidden_dim=emb_sz,
                              rnn_cell=rnn_cell,
                              use_act=st_ds_conf['act'],
                              act_max_layer=st_ds_conf['max_num_layers'],
                              depth_wise_attention=dwa,
                              depth_embedding_type=st_ds_conf['depth_emb'],
                              state_mode=st_ds_conf['act_mode'],
                              )
    model = AdaptiveSeq2Seq(vocab=vocab,
                            encoder=encoder,
                            decoder=decoder,
                            source_embedding=source_embedding,
                            target_embedding=target_embedding,
                            target_namespace='lftokens',
                            start_symbol=START_SYMBOL,
                            eos_symbol=END_SYMBOL,
                            max_decoding_step=st_ds_conf['max_decoding_len'],
                            attention=_get_attention(),
                            act_loss_weight=st_ds_conf['act_loss_weight'],
                            )

    if args.models:
        model.load_state_dict(torch.load(args.models[0]))

    if not args.test or not args.models:
        iterator = BucketIterator(sorting_keys=[("source_tokens", "num_tokens")], batch_size=bsz)
        iterator.index_with(vocab)

        optim = torch.optim.Adam(model.parameters(), lr=config.ADAM_LR, betas=config.ADAM_BETAS, eps=config.ADAM_EPS)

        savepath = os.path.join(config.SNAPSHOT_PATH, args.dataset, 'ada_trans2seq',
                                datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + "--" + args.memo)
        if not os.path.exists(savepath):
            os.makedirs(savepath, mode=0o755)

        trainer = allennlp.training.Trainer(
            model=model,
            optimizer=optim,
            iterator=iterator,
            train_dataset=training_set,
            validation_dataset=validation_set,
            serialization_dir=savepath,
            cuda_device=args.device,
            num_epochs=config.TRAINING_LIMIT,
        )

        trainer.train()

    else:
        testing_set = reader.read(config.DATASETS[args.dataset].test_path)
        model.eval()

        predictor = allennlp.predictors.SimpleSeq2SeqPredictor(model, reader)

        for instance in tqdm.tqdm(testing_set, total=len(testing_set)):
            print('SRC: ', instance.fields['source_tokens'].tokens)
            print('GOLD:', ' '.join(str(x) for x in instance.fields['target_tokens'].tokens[1:-1]))
            del instance.fields['target_tokens']
            output = predictor.predict_instance(instance)
            print('PRED:', ' '.join(output['predicted_tokens']))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass

