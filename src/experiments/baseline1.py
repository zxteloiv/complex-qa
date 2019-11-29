import sys
sys.path.insert(0, '..')
from typing import List, Generator, Tuple, Mapping, Optional
import logging
import os.path
import torch.nn
import random
from collections import defaultdict
from models.matching.re2 import RE2
from torch.nn.utils.rnn import pad_sequence

from trialbot.data import Translator, NSVocabulary, START_SYMBOL, END_SYMBOL, PADDING_TOKEN, DEFAULT_OOV_TOKEN
from trialbot.training import Registry, TrialBot, Events
from trialbot.training.updater import TrainingUpdater, TestingUpdater
from trialbot.training.hparamset import HyperParamSet
from trialbot.utils.move_to_device import move_to_device

from utils.root_finder import find_root
_ROOT = find_root()

@Registry.hparamset()
def atis_none():
    hparams = HyperParamSet.common_settings(_ROOT)
    hparams.emb_sz = 300
    hparams.hidden_size = 150
    hparams.encoder_kernel_size = 3
    hparams.num_classes = 2     # either 0 (true) or 1 (false), only 2 classes
    hparams.num_stacked_block = 3
    hparams.num_stacked_encoder = 2
    hparams.dropout = .5
    hparams.fusion = "full"         # simple, full
    hparams.alignment = "linear"    # identity, linear
    hparams.connection = "aug"      # none, residual, aug
    hparams.prediction = "full"     # simple, full, symmetric
    return hparams

import datasets.atis_rank

@Registry.translator('atis_rank')
class AtisRankTranslator(Translator):
    def __init__(self, max_len: int = 50):
        super().__init__()
        self.max_len = max_len
        self.namespace = ("nl", "lf")   # natural language, and logical form

    def split_nl_sent(self, sent: str):
        return sent.strip().split(" ")

    def split_lf_seq(self, seq: str):
        return seq.strip().split(" ")

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        src, tgt = list(map(example.get, ("src", "tgt")))
        # src is in list, but but tgt is still a sentence
        tgt = self.split_lf_seq(tgt)[:self.max_len]

        ns_nl, ns_lf = self.namespace
        default_toks = [(ns_nl, START_SYMBOL), (ns_nl, END_SYMBOL), (ns_lf, START_SYMBOL), (ns_lf, END_SYMBOL)]
        yield from default_toks

        for tok in src:
            yield ns_nl, tok

        for tok in tgt:
            yield ns_lf, tok

    def to_tensor(self, example):
        assert self.vocab is not None, "vocabulary must be set before assigning ids to instances"

        eid, hyp_rank, src, tgt, hyp = list(map(example.get, ("ex_id", "hyp_rank", "src", "tgt", "hyp")))
        tgt = self.split_lf_seq(tgt)[:self.max_len]
        hyp = self.split_lf_seq(hyp)[:self.max_len]

        ns_nl, ns_lf = self.namespace
        src_toks = torch.tensor([self.vocab.get_token_index(tok, ns_nl) for tok in src])
        tgt_toks = torch.tensor([self.vocab.get_token_index(tok, ns_lf) for tok in tgt])
        hyp_toks = torch.tensor([self.vocab.get_token_index(tok, ns_lf) for tok in hyp])
        instance = {"source_tokens": src_toks, "target_tokens": tgt_toks, "hyp_tokens": hyp_toks,
                    "ex_id": eid, "hyp_rank": hyp_rank}
        return instance

    def batch_tensor(self, tensors: List[Mapping[str, torch.Tensor]]):
        assert len(tensors) > 0
        tensor_list_by_keys = defaultdict(list)
        for instance in tensors:
            for k, tensor in instance.items():
                tensor_list_by_keys[k].append(tensor)

        # # discard batch because every source
        # return tensor_list_by_keys

        padval_by_keys = {
            "source_tokens": self.vocab.get_token_index(PADDING_TOKEN, self.namespace[0]),
            "target_tokens": self.vocab.get_token_index(PADDING_TOKEN, self.namespace[1]),
            "hyp_tokens": self.vocab.get_token_index(PADDING_TOKEN, self.namespace[1]),
        }
        batched_tensor = dict(
            (
                k,
                pad_sequence(tlist, batch_first=True, padding_value=padval_by_keys[k]) \
                    if isinstance(tlist[0], torch.Tensor) else tlist    # pad only lists of tensors
            ) for k, tlist in tensor_list_by_keys.items()
        )
        return batched_tensor

def get_model(hparams, vocab: NSVocabulary):
    model = RE2.get_model(emb_sz=hparams.emb_sz,
                          num_tokens_a=vocab.get_vocab_size('nl'),
                          num_tokens_b=vocab.get_vocab_size('lf'),
                          hid_sz=hparams.hidden_size,
                          enc_kernel_sz=hparams.encoder_kernel_size,
                          num_classes=hparams.num_classes,
                          num_stacked_blocks=hparams.num_stacked_block,
                          num_encoder_layers=hparams.num_stacked_encoder,
                          dropout=hparams.dropout,
                          fusion_mode=hparams.fusion,
                          alignment_mode=hparams.alignment,
                          connection_mode=hparams.connection,
                          prediction_mode=hparams.prediction,
                          use_shared_embedding=False,
                          )
    return model

class Re2TrainingUpdater(TrainingUpdater):
    def update_epoch(self):
        model, optim, iterator = self._models[0], self._optims[0], self._iterators[0]
        if iterator.is_new_epoch:
            self.stop_epoch()

        device = self._device
        model.train()
        optim.zero_grad()

        batch = next(iterator)
        sent_a, sent_b = list(map(batch.get, ("source_tokens", "target_tokens")))

        if device >= 0:
            sent_a = move_to_device(sent_a, device)
            sent_b = move_to_device(sent_b, device)

        batch_size = sent_a.size()[0]
        pos_target = sent_a.new_ones((batch_size,))
        pred_for_pos = model(sent_a, sent_b)
        if self._dry_run:
            return 0

        loss1 = torch.nn.functional.cross_entropy(pred_for_pos, pos_target)
        loss1.backward()

        neg_target = sent_a.new_zeros((batch_size,))
        pred_for_neg = model(sent_a, torch.roll(sent_b, random.randrange(1, batch_size), dims=0))
        loss2 = torch.nn.functional.cross_entropy(pred_for_neg, neg_target)
        loss2.backward()
        optim.step()

        loss = loss1 + loss2
        return {"loss": loss, "prediction": pred_for_pos, "ranking_score": pred_for_pos[:, 1]}

class Re2TestingUpdater(TestingUpdater):
    def update_epoch(self):
        model, iterator, device = self._models[0], self._iterators[0], self._device
        model.eval()
        batch = next(iterator)
        eid, hyp_rank, sent_a, sent_b = list(map(batch.get, ("ex_id", "hyp_rank", "source_tokens", "hyp_tokens")))
        if iterator.is_new_epoch:
            self.stop_epoch()

        if device >= 0:
            sent_a = move_to_device(sent_a, device)
            sent_b = move_to_device(sent_b, device)

        output = model(sent_a, sent_b)
        output = torch.log_softmax(output, dim=-1)
        correct_score = output[:, 1]
        return {"prediction": output, "ranking_score": correct_score, "ex_id": eid, "hyp_rank": hyp_rank}

def main():
    import sys
    args = sys.argv[1:]
    if '--dataset' not in sys.argv:
        args += ['--dataset', 'atis_none_hyp']
    if '--translator' not in sys.argv:
        args += ['--translator', 'atis_rank']

    parser = TrialBot.get_default_parser()
    args = parser.parse_args(args)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.INFO)

    bot = TrialBot(trial_name="reranking_baseline1", get_model_func=get_model, args=args)
    if args.test:
        import trialbot
        new_engine = trialbot.training.trial_bot.Engine()
        new_engine.register_events(*Events)
        bot._engine = new_engine

        @bot.attach_extension(Events.ITERATION_COMPLETED)
        def print_output(bot: TrialBot):
            import json
            output = bot.state.output
            if output is None:
                return

            output_keys = ("ex_id", "hyp_rank", "ranking_score")
            for eid, hyp_rank, score in zip(*map(output.get, output_keys)):
                print(json.dumps(dict(zip(output_keys, (eid, hyp_rank, score.item())))))

        bot.updater = Re2TestingUpdater.from_bot(bot)
    else:
        from trialbot.training.extensions import every_epoch_model_saver

        def output_inspect(bot: TrialBot, keys):
            iteration = bot.state.iteration
            if iteration % 4 != 0:
                return

            output = bot.state.output
            bot.logger.info(", ".join(f"{k}={v}" for k, v in zip(keys, map(output.get, keys))))

        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)
        # bot.add_event_handler(Events.ITERATION_COMPLETED, output_inspect, 100, keys=["loss"])
        bot.updater = Re2TrainingUpdater.from_bot(bot)
    bot.run()

if __name__ == '__main__':
    main()

