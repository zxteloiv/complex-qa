import os.path
import datetime
import torch
import torch.nn

import allennlp.data
from allennlp.data.iterators import BucketIterator

import allennlp.training

import logging
import allennlp
import allennlp.common
import allennlp.models
import allennlp.modules
import allennlp.predictors
from allennlp.modules.attention import BilinearAttention, DotProductAttention
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper

import config
import pickle

import data_adapter
import training.exp_runner.opt_parser
from models.base_s2s.base_seq2seq import BaseSeq2Seq
from models.transformer.multi_head_attention import SingleTokenMHAttentionWrapper, GeneralMultiHeadAttention
from utils.nn import AllenNLPAttentionWrapper
from models.transformer.encoder import TransformerEncoder
from models.modules.universal_hidden_state_wrapper import UniversalHiddenStateWrapper, RNNType
from models.modules.stacked_rnn_cell import StackedLSTMCell, StackedGRUCell
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from models.modules.stacked_encoder import StackedEncoder

def main():
    parser = training.exp_runner.opt_parser.get_trainer_opt_parser()
    parser.add_argument('models', nargs='*', help='pretrained models for the same setting')
    parser.add_argument('--test', action="store_true", help='use testing mode')
    parser.add_argument('--hparamset', help="available hyper-parameters")
    parser.add_argument('--from-hparamset-dump', help='read hyperparameters from the dump file, to reproduce')
    parser.add_argument('--list-hparamset', action='store_true')
    parser.add_argument('--snapshot-dir', help="snapshot dir if continues")
    parser.add_argument('--dataset', choices=config.DATASETS.keys())
    parser.add_argument('--data-reader', choices=data_adapter.DATA_READERS.keys())

    args = parser.parse_args()

    if args.list_hparamset:
        import json
        print(json.dumps(config.SETTINGS, indent=4))
        return

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    if args.from_hparamset_dump:
        conf = pickle.load(open(args.from_hparamset_dump))
    else:
        conf = getattr(config, args.hparamset)()

    if args.device >= 0:
        conf.DEVICE = args.device

    if conf.DEVICE < 0:
        run_model(args, conf)
    else:
        with torch.cuda.device(conf.DEVICE):
            run_model(args, conf)

def run_model(args, conf):
    logging.debug(str(args))
    logging.debug(str(conf))
    reader = data_adapter.DATA_READERS[args.data_reader]()
    dataset_path = config.DATASETS[args.dataset]
    training_set = reader.read(dataset_path.train_path)
    try:
        validation_set = reader.read(config.DATASETS[args.dataset].dev_path)
    except:
        validation_set = None

    vocab = allennlp.data.Vocabulary.from_instances(training_set)
    model = get_model(vocab, conf)

    if args.models:
        model.load_state_dict(torch.load(args.models[0], map_location='cpu'))

    if conf.DEVICE >= 0:
        model = model.cuda(conf.DEVICE)

    if not args.test or not args.models:
        iterator = BucketIterator(sorting_keys=[("source_tokens", "num_tokens")], batch_size=conf.batch_sz)
        iterator.index_with(vocab)

        optim = torch.optim.Adam(model.parameters(), lr=conf.ADAM_LR, betas=conf.ADAM_BETAS, eps=conf.ADAM_EPS)

        savepath = args.snapshot_dir if args.snapshot_dir else (os.path.join(
            conf.SNAPSHOT_PATH,
            args.dataset,
            'base_s2s',
            datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + ('-' + args.memo if args.memo else '')
        ))
        if not os.path.exists(savepath):
            os.makedirs(savepath, mode=0o755)

        trainer = allennlp.training.Trainer(
            model=model,
            optimizer=optim,
            iterator=iterator,
            train_dataset=training_set,
            validation_dataset=validation_set,
            serialization_dir=savepath,
            cuda_device=conf.DEVICE,
            num_epochs=conf.TRAINING_LIMIT,
            grad_clipping=conf.GRAD_CLIPPING,
        )

        trainer.train()

    else:
        testing_set = reader.read(config.DATASETS[args.dataset].test_path)
        model.eval()

        if config.DEVICE > -1:
            model = model.cuda(config.DEVICE)

        predictor = allennlp.predictors.Seq2SeqPredictor(model, reader)

        for instance in testing_set:
            print('SRC: ', ' '.join(str(tok) for tok in instance.fields['source_tokens'].tokens))
            print('GOLD:', ' '.join(str(x) for x in instance.fields['target_tokens'].tokens[1:-1]))
            del instance.fields['target_tokens']
            output = predictor.predict_instance(instance)
            print('PRED:', ' '.join(output['predicted_tokens']))

def get_model(vocab, conf):
    emb_sz = conf.emb_sz

    source_embedding = allennlp.modules.Embedding(num_embeddings=vocab.get_vocab_size('nltokens'),
                                                  embedding_dim=emb_sz)
    target_embedding = allennlp.modules.Embedding(num_embeddings=vocab.get_vocab_size('lftokens'),
                                                  embedding_dim=emb_sz)

    if conf.encoder == 'lstm':
        encoder = StackedEncoder([
            PytorchSeq2SeqWrapper(torch.nn.LSTM(emb_sz, emb_sz, batch_first=True))
            for _ in range(conf.num_enc_layers)
        ], emb_sz, emb_sz, input_dropout=conf.intermediate_dropout)
    elif conf.encoder == 'bilstm':
        encoder = StackedEncoder([
            PytorchSeq2SeqWrapper(torch.nn.LSTM(emb_sz, emb_sz, batch_first=True, bidirectional=True))
            for _ in range(conf.num_enc_layers)
        ], emb_sz, emb_sz, input_dropout=conf.intermediate_dropout)
    elif conf.encoder == 'transformer':
        encoder = StackedEncoder([
            TransformerEncoder(input_dim=emb_sz,
                               num_layers=1,
                               num_heads=conf.num_heads,
                               feedforward_hidden_dim=emb_sz,
                               feedforward_dropout=conf.feedforward_dropout,
                               residual_dropout=conf.residual_dropout,
                               attention_dropout=conf.attention_dropout,
                               )
            for _ in range(conf.num_enc_layers)
        ], emb_sz, emb_sz, input_dropout=conf.intermediate_dropout)
    else:
        assert False

    enc_out_dim = encoder.get_output_dim()
    dec_out_dim = emb_sz

    dec_hist_attn = get_attention(conf, conf.dec_hist_attn)
    enc_attn = get_attention(conf, conf.enc_attn)
    if conf.enc_attn == 'dot_product':
        assert enc_out_dim == dec_out_dim, "encoder hidden states must be able to multiply with decoder output"

    def sum_attn_dims(attns, dims):
        return sum(dim for attn, dim in zip(attns, dims) if attn is not None)

    if conf.concat_attn_to_dec_input:
        dec_in_dim = dec_out_dim + sum_attn_dims([enc_attn, dec_hist_attn], [enc_out_dim, dec_out_dim])
    else:
        dec_in_dim = dec_out_dim
    rnn_cell = get_rnn_cell(conf, dec_in_dim, dec_out_dim)

    if conf.concat_attn_to_dec_input:
        proj_in_dim = dec_out_dim + sum_attn_dims([enc_attn, dec_hist_attn], [enc_out_dim, dec_out_dim])
    else:
        proj_in_dim = dec_out_dim

    word_proj = torch.nn.Linear(proj_in_dim, vocab.get_vocab_size('lftokens'))

    model = BaseSeq2Seq(vocab=vocab,
                        encoder=encoder,
                        decoder=rnn_cell,
                        word_projection=word_proj,
                        source_embedding=source_embedding,
                        target_embedding=target_embedding,
                        target_namespace='lftokens',
                        start_symbol=START_SYMBOL,
                        eos_symbol=END_SYMBOL,
                        max_decoding_step=conf.max_decoding_len,
                        enc_attention=enc_attn,
                        dec_hist_attn=dec_hist_attn,
                        intermediate_dropout=conf.intermediate_dropout,
                        concat_attn_to_dec_input=conf.concat_attn_to_dec_input,
                        )
    return model

def get_rnn_cell(conf, input_dim: int, hidden_dim: int):
    cell_type = conf.decoder
    if cell_type == "lstm":
        return UniversalHiddenStateWrapper(RNNType.LSTM(input_dim, hidden_dim))
    elif cell_type == "gru":
        return UniversalHiddenStateWrapper(RNNType.GRU(input_dim, hidden_dim))
    elif cell_type == "ind_rnn":
        return UniversalHiddenStateWrapper(RNNType.IndRNN(input_dim, hidden_dim))
    elif cell_type == "rnn":
        return UniversalHiddenStateWrapper(RNNType.VanillaRNN(input_dim, hidden_dim))
    elif cell_type == 'n_lstm':
        n_layer = conf.dec_cell_height
        return StackedLSTMCell(input_dim, hidden_dim, n_layer, conf.intermediate_dropout)
    elif cell_type == 'n_gru':
        n_layer = conf.dec_cell_height
        return StackedGRUCell(input_dim, hidden_dim, n_layer, conf.intermediate_dropout)
    else:
        raise ValueError(f"RNN type of {cell_type} not found.")

def get_attention(conf, attn_type):
    emb_sz = conf.emb_sz   # dim for both the decoder output and the encoder output
    attn_type = attn_type.lower()
    if attn_type == "bilinear":
        attn = BilinearAttention(vector_dim=emb_sz, matrix_dim=emb_sz)
        attn = AllenNLPAttentionWrapper(attn, conf.attention_dropout)
    elif attn_type == "dot_product":
        attn = DotProductAttention()
        attn = AllenNLPAttentionWrapper(attn, conf.attention_dropout)
    elif attn_type == "multihead":
        attn = GeneralMultiHeadAttention(num_heads=conf.num_heads,
                                         input_dim=emb_sz,
                                         total_attention_dim=emb_sz,
                                         total_value_dim=emb_sz,
                                         attend_to_dim=emb_sz,
                                         output_dim=emb_sz,
                                         attention_dropout=conf.attention_dropout,
                                         use_future_blinding=False,
                                         )
        attn = SingleTokenMHAttentionWrapper(attn)
    elif attn_type == "none":
        attn = None
    else:
        assert False

    return attn

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass


