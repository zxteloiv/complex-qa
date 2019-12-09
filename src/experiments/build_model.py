from trialbot.data import NSVocabulary, PADDING_TOKEN
import torch
from torch import nn

def get_re2_model(hparams, vocab: NSVocabulary):
    from models.matching.re2 import RE2
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

def get_re2_variant(hparams, vocab: NSVocabulary):
    from models.modules.stacked_encoder import StackedEncoder
    from models.matching.mha_encoder import MHAEncoder
    from models.matching.re2 import RE2
    from models.matching.re2_modules import Re2Block, Re2Prediction, Re2Pooling, Re2Conn
    from models.matching.re2_modules import Re2Alignment, Re2Fusion

    emb_sz, hid_sz, dropout = hparams.emb_sz, hparams.hidden_size, hparams.dropout
    embedding_a = torch.nn.Embedding(vocab.get_vocab_size('nl'), emb_sz)
    embedding_b = torch.nn.Embedding(vocab.get_vocab_size('lf'), emb_sz)

    conn: Re2Conn = Re2Conn(hparams.connection, emb_sz, hid_sz)
    conn_out_sz = conn.get_output_size()

    # the input to predict is exactly the output of fusion, with the hidden size
    pred = Re2Prediction(hparams.prediction, inp_sz=hid_sz, hid_sz=hid_sz,
                         num_classes=hparams.num_classes, dropout=dropout)
    pooling = Re2Pooling()

    def _encoder(inp_sz):
        if hasattr(hparams, "encoder") and hparams.encoder == "lstm":
            from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
            return PytorchSeq2SeqWrapper(torch.nn.LSTM(inp_sz, hid_sz, hparams.num_stacked_encoder,
                                                       batch_first=True, dropout=dropout, bidirectional=False))
        elif hasattr(hparams, "encoder") and hparams.encoder == "bilstm":
            from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
            return PytorchSeq2SeqWrapper(torch.nn.LSTM(inp_sz, hid_sz // 2, hparams.num_stacked_encoder,
                                                       batch_first=True, dropout=dropout, bidirectional=True))
        else:
            return StackedEncoder([
                MHAEncoder(inp_sz if j == 0 else hid_sz, hid_sz, hparams.num_heads, dropout)
                for j in range(hparams.num_stacked_encoder)
            ], inp_sz, hid_sz, dropout, output_every_layer=False)

    enc_inp_sz = lambda i: emb_sz if i == 0 else conn_out_sz
    blocks = torch.nn.ModuleList([
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

def get_giant_model(hparams, vocab: NSVocabulary):
    from models.matching.giant_ranker import GiantRanker
    from models.matching.re2 import RE2
    from models.matching.seq2seq_modeling import Seq2SeqModeling
    from models.matching.seq_modeling import SeqModeling, PytorchSeq2SeqWrapper
    from allennlp.modules.matrix_attention import BilinearMatrixAttention

    re2: RE2 = get_re2_variant(hparams, vocab)
    emb_sz, hid_sz, dropout = hparams.emb_sz, hparams.hidden_size, hparams.dropout
    RNNWrapper = PytorchSeq2SeqWrapper

    a2b = Seq2SeqModeling(a_embedding=re2.a_emb,
                          b_embedding=re2.b_emb,
                          encoder=RNNWrapper(nn.LSTM(emb_sz, hid_sz, num_layers=2, batch_first=True,
                                                     dropout=dropout), stateful=True),
                          decoder=RNNWrapper(nn.LSTM(emb_sz, hid_sz, num_layers=2, batch_first=True,
                                                     dropout=dropout)),
                          a_padding=re2.padding_val_a,
                          b_padding=re2.padding_val_b,
                          prediction=nn.Linear(hid_sz * 2, vocab.get_vocab_size('lf')),
                          attention=BilinearMatrixAttention(hid_sz, hid_sz),
                          )

    b2a = Seq2SeqModeling(a_embedding=re2.b_emb,
                          b_embedding=re2.a_emb,
                          encoder=RNNWrapper(nn.LSTM(emb_sz, hid_sz, num_layers=2, batch_first=True,
                                                     dropout=dropout), stateful=True),
                          decoder=RNNWrapper(nn.LSTM(emb_sz, hid_sz, num_layers=2, batch_first=True,
                                                     dropout=dropout)),
                          a_padding=re2.padding_val_b,
                          b_padding=re2.padding_val_a,
                          prediction=nn.Linear(hid_sz * 2, vocab.get_vocab_size('nl')),
                          attention=BilinearMatrixAttention(hid_sz, hid_sz),
                          )

    a_seq = SeqModeling(embedding=re2.a_emb,
                        encoder=RNNWrapper(nn.LSTM(emb_sz, hid_sz, num_layers=2, batch_first=True,
                                                   dropout=dropout)),
                        padding=re2.padding_val_a,
                        prediction=nn.Linear(hid_sz, vocab.get_vocab_size('nl')),
                        attention=BilinearMatrixAttention(hid_sz, hid_sz),
                        )

    b_seq = SeqModeling(embedding=re2.b_emb,
                        encoder=RNNWrapper(nn.LSTM(emb_sz, hid_sz, num_layers=2, batch_first=True,
                                                   dropout=dropout)),
                        padding=re2.padding_val_b,
                        prediction=nn.Linear(hid_sz, vocab.get_vocab_size('lf')),
                        attention=BilinearMatrixAttention(hid_sz, hid_sz),
                        )

    model = GiantRanker(re2.a_emb, re2.b_emb, re2, a2b, b2a, a_seq, b_seq, re2.padding_val_a, re2.padding_val_b)
    return model
