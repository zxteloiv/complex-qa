from typing import Callable, Optional
import logging
import torch
import pickle
import config
import os.path
from datetime import datetime

import data_adapter
import allennlp.data
import allennlp.predictors
from allennlp.data.iterators import BucketIterator

from .opt_parser import get_common_opt_parser
logger = logging.getLogger(__name__)

class ExperimentRunner:
    def __init__(self,
                 exp_name="default_savedir",
                 get_model_func: Optional[Callable] = None,
                 reader=None,
                 dataset_path=None,
                 vocab=None):
        self.parser = get_common_opt_parser()
        self.parser.add_argument('models', nargs='*', help='pretrained models for the same setting')
        self.name = exp_name
        self.reader = reader
        self.dataset_path = dataset_path
        self.vocab = vocab
        self.train_set = None
        self.dev_set = None
        self.test_set = None
        self.args = self.parser.parse_args()
        self.hparams = config.SETTINGS[self.args.hparamset]() if self.args.hparamset else None
        self.fn_get_model = get_model_func

    def run(self):
        args, hparams = self.args, self.hparams

        if args.list_hparamset:
            import json
            print(json.dumps(list(config.SETTINGS.keys()), indent=4))
            return

        # logging args
        if args.quiet:
            logger.setLevel(logging.WARNING)
        elif args.debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        if args.device >= 0:
            hparams.DEVICE = args.device

        self.init_components(args, hparams)
        model = self.get_model(hparams, self.vocab)

        if hparams.DEVICE >= 0:
            model = model.cuda(hparams.DEVICE)

        if args.test:
            self.eval(args, hparams, model)
        else:
            self.train(args, hparams, model)

    def get_model(self, hparams, vocab):
        if self.fn_get_model is not None:
            return self.fn_get_model(hparams, vocab)
        else:
            raise NotImplemented

    def init_components(self, args, hparams):
        if self.dataset_path is None:
            self.dataset_path = config.DATASETS[args.dataset]

        if self.reader is None:
            self.reader = data_adapter.DATA_READERS[args.data_reader]()

        if self.vocab is None:
            if self.train_set is None:
                self.train_set = self.reader.read(self.dataset_path.train_path)
            self.vocab = allennlp.data.Vocabulary.from_instances(self.train_set, min_count={"tokens": 3})
            logger.info(str(self.vocab))

    def train(self, args, hparams, model):
        try:
            validation_set = self.reader.read(self.dataset_path.dev_path)
        except:
            validation_set = None
        iterator = BucketIterator(sorting_keys=[("source_tokens", "num_tokens")], batch_size=hparams.batch_sz)
        iterator.index_with(self.vocab)
        optim = torch.optim.Adam(model.parameters(), hparams.ADAM_LR, hparams.ADAM_BETAS)

        savepath = args.snapshot_dir if args.snapshot_dir else (os.path.join(
            hparams.SNAPSHOT_PATH,
            args.dataset,
            self.name,
            datetime.now().strftime('%Y%m%d-%H%M%S') + ('-' + args.memo if args.memo else '')
        ))
        if not os.path.exists(savepath):
            os.makedirs(savepath, mode=0o755)

        trainer = allennlp.training.Trainer(
            model=model,
            optimizer=optim,
            iterator=iterator,
            train_dataset=self.train_set,
            validation_dataset=validation_set,
            serialization_dir=savepath,
            cuda_device=hparams.DEVICE,
            num_epochs=hparams.TRAINING_LIMIT,
        )

        if hparams.DEVICE < 0:
            trainer.train()
        else:
            with torch.cuda.device(hparams.DEVICE):
                trainer.train()

    def eval(self, args, hparams, model):
        testing_set = self.reader.read(self.dataset_path.test_path)
        model.eval()

        predictor = allennlp.predictors.Seq2SeqPredictor(model, self.reader)

        for instance in testing_set:
            print('SRC: ', ' '.join(str(tok) for tok in instance.fields['source_tokens'].tokens))
            print('GOLD:', ' '.join(str(x) for x in instance.fields['target_tokens'].tokens[1:-1]))
            del instance.fields['target_tokens']
            output = predictor.predict_instance(instance)
            print('PRED:', ' '.join(output['predicted_tokens']))


