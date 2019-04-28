import os.path
import logging
import datetime
import config
import utils.opt_parser
import data_adapter
import torch
import allennlp
import allennlp.data
import allennlp.modules
import allennlp.models
import allennlp.training
import allennlp.predictors
import pickle

from allennlp.data.iterators import BucketIterator
from models.parallel_seq2seq import ParallelSeq2Seq
from models.transformer.encoder import TransformerEncoder
from models.transformer.decoder import TransformerDecoder
from allennlp.common.util import START_SYMBOL, END_SYMBOL

def main():
    parser = utils.opt_parser.get_trainer_opt_parser()
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
        validation_set = reader.read(dataset_path.dev_path)
    except:
        validation_set = None

    vocab = allennlp.data.Vocabulary.from_instances(training_set)

    encoder = TransformerEncoder(input_dim=conf.emb_sz,
                                 num_layers=conf.num_layers,
                                 num_heads=conf.num_heads,
                                 feedforward_hidden_dim=conf.emb_sz,)
    decoder = TransformerDecoder(input_dim=conf.emb_sz,
                                 num_layers=conf.num_layers,
                                 num_heads=conf.num_heads,
                                 feedforward_hidden_dim=conf.emb_sz,
                                 feedforward_dropout=conf.feedforward_dropout,)
    source_embedding = allennlp.modules.Embedding(num_embeddings=vocab.get_vocab_size('src_tokens'),
                                                  embedding_dim=conf.emb_sz)
    target_embedding = allennlp.modules.Embedding(num_embeddings=vocab.get_vocab_size('tgt_tokens'),
                                                  embedding_dim=conf.emb_sz)
    model = ParallelSeq2Seq(vocab=vocab,
                            encoder=encoder,
                            decoder=decoder,
                            source_embedding=source_embedding,
                            target_embedding=target_embedding,
                            target_namespace='tgt_tokens',
                            start_symbol=START_SYMBOL,
                            eos_symbol=END_SYMBOL,
                            max_decoding_step=conf.max_decoding_len,
                            )
    if conf.DEVICE >= 0:
        model = model.cuda(conf.DEVICE)

    if args.models:
        model.load_state_dict(torch.load(args.models[0]))

    if not args.test or not args.models:
        iterator = BucketIterator(sorting_keys=[("source_tokens", "num_tokens")], batch_size=conf.batch_sz)
        iterator.index_with(vocab)

        optim = torch.optim.Adam(model.parameters())

        savepath = args.snapshot_dir if args.snapshot_dir else (os.path.join(
            conf.SNAPSHOT_PATH,
            args.dataset,
            'transformer',
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
        )

        trainer.train()

    else:
        testing_set = reader.read(config.DATASETS[args.dataset].test_path)
        model.eval()

        predictor = allennlp.predictors.Seq2SeqPredictor(model, reader)

        for instance in testing_set:
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