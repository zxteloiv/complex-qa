from typing import Callable, Optional, Type, Dict
import logging
import torch
import config
import os.path
from datetime import datetime

from data_adapter.dataset import Dataset
from data_adapter.ns_vocabulary import NSVocabulary
import data_adapter.general_datasets
import data_adapter.translators.weibo_keyword_translator
import ignite
from data_adapter.iterators.random_iterator import RandomIterator

from .opt_parser import get_common_opt_parser
from .trial_registry import Registry

logger = logging.getLogger('TrialBot')


class TrialBot:
    def __init__(self,
                 args = None,
                 trial_name="default_savedir",
                 get_model_func: Optional[Callable] = None,
                 ):
        if args is None:
            parser = TrialBot.get_default_parser()
            args = parser.parse_args()

        self.args = args
        self.name = trial_name

        # self._translator = None
        # self._vocab = None
        self._train_set = None
        self._dev_set = None
        self._test_set = None
        # self._hparams = None
        # self.fn_get_model = get_model_func

    @staticmethod
    def get_default_parser():
        parser = get_common_opt_parser()
        parser.add_argument('models', nargs='*', help='pretrained models for the same setting')
        parser.add_argument('--vocab-dump', help="the file path to save and load the vocab obj")
        return parser

    def run(self):
        """
        Start a trial directly.
        """
        args = self.args
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

        hparams = Registry.get_hparamset(args.hparamset)
        train_set, dev_set, test_set = Registry.get_dataset(args.dataset)
        translator: data_adapter.translator = Registry.get_translator(args.reader)

        if args.vocab_dump and os.path.exists(args.vocab_dump):
            vocab = data_adapter.ns_vocabulary.NSVocabulary.from_files(self.args.vocab_dump)
        else:
            vocab = self.init_vocab(train_set, translator)

        if args.vocab_dump:
            os.makedirs(args.vocab_dump)
            vocab.save_to_files(args.vocab_dump)
        logger.info(str(vocab))

        model = self.get_model(hparams, vocab)
        if args.models:
            model.load_state_dict(torch.load(args.models[0]))

        if args.device >= 0:
            model = model.cuda(args.device)

        if args.test:
            self.test(args, hparams, model)
        else:
            self.train(args, hparams, model)


    def init_vocab(self,
                   dataset: data_adapter.dataset.Dataset,
                   translator: data_adapter.translator.Translator):
        # count for a counter
        counter: Dict[str, Dict[str, int]] = dict()
        for example in iter(dataset):
            for namespace, w in translator.generate_namespace_tokens(example):
                if namespace not in counter:
                    counter[namespace] = dict()
                counter[namespace][w] += 1

        vocab = NSVocabulary(counter, min_count={"tokens": 3})
        return vocab


    def get_model(self, hparams, vocab):
        if self.fn_get_model is not None:
            return self.fn_get_model(hparams, vocab)
        else:
            raise NotImplemented

    def train(self, args, hparams, model):
        try:
            if self.train_set is None:
                self.train_set = self.reader.read(self.dataset_path.train_path)
            validation_set = self.reader.read(self.dataset_path.dev_path)
        except:
            validation_set = None

        iterator = RandomIterator(sorting_keys=[("source_tokens", "num_tokens")], batch_size=hparams.batch_sz,)
        optim = torch.optim.Adam(model.parameters(), hparams.ADAM_LR, hparams.ADAM_BETAS)

        savepath = args.snapshot_dir if args.snapshot_dir else (os.path.join(
            hparams.SNAPSHOT_PATH,
            args.dataset,
            self.name,
            datetime.now().strftime('%Y%m%d-%H%M%S') + ('-' + args.memo if args.memo else '')
        ))
        if not os.path.exists(savepath):
            os.makedirs(savepath, mode=0o755)
        vocab_path = os.path.join(savepath, 'vocab')
        if not os.path.exists(vocab_path):
            os.makedirs(vocab_path, mode=0o755)
            self.vocab.save_to_files(vocab_path)

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

    def test(self, args, hparams, model):
        testing_set = self.reader.read(self.dataset_path.test_path)
        model.eval()

        predictor = allennlp.predictors.Seq2SeqPredictor(model, self.reader)

        for instance in testing_set:
            print('SRC: ', ' '.join(str(tok) for tok in instance.fields['source_tokens'].tokens))
            print('GOLD:', ' '.join(str(x) for x in instance.fields['target_tokens'].tokens[1:-1]))
            del instance.fields['target_tokens']
            output = predictor.predict_instance(instance)
            sent = output['predicted_tokens']
            if len(sent) == 1:
                print('PRED:', ' '.join(output['predicted_tokens'][0]))
            else:
                for i, beam in enumerate(sent):
                    print('BEAM%d:' % i, ' '.join(beam))



