import sys
sys.path.insert(0, '..')
from typing import List, Generator, Tuple, Mapping, Optional
import logging
import torch
import torch.nn as nn
import random
from models.matching.re2 import RE2

from trialbot.data import NSVocabulary, PADDING_TOKEN
from trialbot.training import Registry, TrialBot, Events
from trialbot.training.updater import TrainingUpdater, TestingUpdater
from trialbot.training.hparamset import HyperParamSet
from trialbot.utils.move_to_device import move_to_device

from utils.root_finder import find_root
_ROOT = find_root()

@Registry.hparamset()
def atis_five_mha_enc():
    hparams = HyperParamSet.common_settings(_ROOT)
    hparams.emb_sz = 300
    hparams.hidden_size = 150
    hparams.num_heads = 10

    hparams.num_classes = 2     # either 0 (true) or 1 (false), only 2 classes
    hparams.num_stacked_block = 3
    hparams.num_stacked_encoder = 2
    hparams.dropout = .5
    hparams.fusion = "full"         # simple, full
    hparams.alignment = "linear"    # identity, linear
    hparams.connection = "aug"      # none, residual, aug
    hparams.prediction = "full"     # simple, full, symmetric
    return hparams

@Registry.hparamset()
def atis_five_lstm_enc():
    hparams = atis_five_mha_enc()
    hparams.encoder = "lstm"
    return hparams

import datasets.atis_rank
import datasets.atis_rank_translator

def get_model(hparams, vocab: NSVocabulary):
    # RE2 requires an encoder which
    # maps (sent in (batch, N, inp_sz), and mask in (batch, N))
    # to (sent_prime in (batch, N, hidden)
    from models.modules.stacked_encoder import StackedEncoder
    from models.matching.mha_encoder import MHAEncoder
    from models.matching.re2_modules import Re2Block, Re2Prediction, Re2Pooling, Re2Conn
    from models.matching.re2_modules import Re2Alignment, Re2Fusion

    emb_sz, hid_sz, dropout = hparams.emb_sz, hparams.hidden_size, hparams.dropout
    embedding_a = nn.Embedding(vocab.get_vocab_size('nl'), emb_sz)
    embedding_b = nn.Embedding(vocab.get_vocab_size('lf'), emb_sz)

    conn: Re2Conn = Re2Conn(hparams.connection, emb_sz, hid_sz)
    conn_out_sz = conn.get_output_size()

    # the input to predict is exactly the output of fusion, with the hidden size
    pred = Re2Prediction(hparams.prediction, inp_sz=hid_sz, hid_sz=hid_sz,
                         num_classes=hparams.num_classes, dropout=dropout)
    pooling = Re2Pooling()


    def _encoder(inp_sz):
        if hasattr(hparams, "encoder") and hparams.encoder == "lstm":
            from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
            return PytorchSeq2SeqWrapper(nn.LSTM(inp_sz, hid_sz, hparams.num_stacked_encoder,
                                                 batch_first=True, dropout=dropout, bidirectional=False))
        else:
            return StackedEncoder([
                MHAEncoder(inp_sz if j == 0 else hid_sz, hid_sz, hparams.num_heads, dropout)
                for j in range(hparams.num_stacked_encoder)
            ], inp_sz, hid_sz, dropout, output_every_layer=False)

    enc_inp_sz = lambda i: emb_sz if i == 0 else conn_out_sz
    blocks = nn.ModuleList([
        Re2Block(
            _encoder(enc_inp_sz(i)),
            _encoder(enc_inp_sz(i)),
            Re2Fusion(hid_sz + enc_inp_sz(i), hid_sz, hparams.fusion == "full", dropout),
            Re2Fusion(hid_sz + enc_inp_sz(i), hid_sz, hparams.fusion == "full", dropout),
            Re2Alignment(hid_sz + enc_inp_sz(i), hid_sz, hparams.alignment),
            dropout=dropout,
        )
        for i in range(hparams.num_stacked_block)
    ])

    model = RE2(embedding_a, embedding_b, blocks, pooling, pooling, conn, pred,
                vocab.get_token_index(PADDING_TOKEN, 'nl'),
                vocab.get_token_index(PADDING_TOKEN, 'lf'))
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
        sent_a, sent_b, label = list(map(batch.get, ("source_tokens", "hyp_tokens", "hyp_label")))

        if device >= 0:
            sent_a = move_to_device(sent_a, device)
            sent_b = move_to_device(sent_b, device)
            label = move_to_device(label, device)

        logits = model(sent_a, sent_b)
        if self._dry_run:
            return 0

        loss = torch.nn.functional.cross_entropy(logits, label)
        loss.backward()
        optim.step()
        return {"loss": loss, "prediction": logits}

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
        args += ['--dataset', 'atis_five_hyp']
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

    bot = TrialBot(trial_name="reranking_baseline2_transformer", get_model_func=get_model, args=args)
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
        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)
        bot.updater = Re2TrainingUpdater.from_bot(bot)
    bot.run()

if __name__ == '__main__':
    main()


