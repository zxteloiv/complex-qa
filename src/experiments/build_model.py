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
    from models.modules.embedding_dropout import SeqEmbeddingDropoutWrapper
    from models.matching.mha_encoder import MHAEncoder
    from models.matching.re2 import RE2
    from models.matching.re2_modules import Re2Block, Re2Prediction, Re2Conn, Re2Fusion, Re2Alignment, Re2Pooling
    from models.matching.re2_modules import NeoRe2Pooling, NeoFusion

    emb_sz, hid_sz, dropout = hparams.emb_sz, hparams.hidden_size, hparams.dropout
    embedding_a = torch.nn.Embedding(vocab.get_vocab_size('nl'), emb_sz)
    embedding_b = torch.nn.Embedding(vocab.get_vocab_size('lf'), emb_sz)

    d_dropout = hparams.discrete_dropout if hasattr(hparams, "discrete_dropout") else 0.
    i_dropout = hparams.dropout if hasattr(hparams, "dropout") else 0.
    embedding_a = SeqEmbeddingDropoutWrapper(embedding_a, d_dropout, i_dropout)
    embedding_b = SeqEmbeddingDropoutWrapper(embedding_b, d_dropout, i_dropout)

    conn: Re2Conn = Re2Conn(hparams.connection, emb_sz, hid_sz)
    conn_out_sz = conn.get_output_size()

    if hasattr(hparams, "pooling") and hparams.pooling == "neo":
        # neo fusion outputs 3times larger embedding, contains max, mean, and std pooling
        pooling = NeoRe2Pooling()
        pred_inp = hid_sz * 3
    else:
        # common fusion uses only max pooling
        pooling = Re2Pooling()
        pred_inp = hid_sz

    pred = Re2Prediction(hparams.prediction, inp_sz=pred_inp, hid_sz=hid_sz,
                         num_classes=hparams.num_classes, dropout=dropout, activation=nn.SELU())

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

    fusion_cls = NeoFusion if hasattr(hparams, "fusion") and hparams.fusion == "neo" else Re2Fusion

    enc_inp_sz = lambda i: emb_sz if i == 0 else conn_out_sz
    blocks = torch.nn.ModuleList([
        Re2Block(
            _encoder(enc_inp_sz(i)),
            _encoder(enc_inp_sz(i)),
            fusion_cls(hid_sz + enc_inp_sz(i), hid_sz, hparams.fusion == "full", dropout),
            fusion_cls(hid_sz + enc_inp_sz(i), hid_sz, hparams.fusion == "full", dropout),
            Re2Alignment(hid_sz + enc_inp_sz(i), hid_sz, hparams.alignment, nn.SELU()),
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
    get_rnn = lambda stateful: RNNWrapper(nn.LSTM(emb_sz, hid_sz, num_layers=hparams.num_stacked_encoder,
                                                  batch_first=True, dropout=hparams.dropout), stateful=stateful)

    a2b = Seq2SeqModeling(a_embedding=re2.a_emb,
                          b_embedding=re2.b_emb,
                          encoder=get_rnn(stateful=True),
                          decoder=get_rnn(stateful=False),
                          a_padding=re2.padding_val_a,
                          b_padding=re2.padding_val_b,
                          prediction=nn.Linear(hid_sz * 2, vocab.get_vocab_size('lf')),
                          attention=BilinearMatrixAttention(hid_sz, hid_sz),
                          )

    b2a = Seq2SeqModeling(a_embedding=re2.b_emb,
                          b_embedding=re2.a_emb,
                          encoder=get_rnn(stateful=True),
                          decoder=get_rnn(stateful=False),
                          a_padding=re2.padding_val_b,
                          b_padding=re2.padding_val_a,
                          prediction=nn.Linear(hid_sz * 2, vocab.get_vocab_size('nl')),
                          attention=BilinearMatrixAttention(hid_sz, hid_sz),
                          )

    a_seq = SeqModeling(embedding=re2.a_emb,
                        encoder=get_rnn(stateful=False),
                        padding=re2.padding_val_a,
                        prediction=nn.Linear(hid_sz * 2, vocab.get_vocab_size('nl')),
                        attention=BilinearMatrixAttention(hid_sz, hid_sz),
                        )

    b_seq = SeqModeling(embedding=re2.b_emb,
                        encoder=get_rnn(stateful=False),
                        padding=re2.padding_val_b,
                        prediction=nn.Linear(hid_sz * 2, vocab.get_vocab_size('lf')),
                        attention=BilinearMatrixAttention(hid_sz, hid_sz),
                        )

    model = GiantRanker(re2.a_emb, re2.b_emb, re2, a2b, b2a, a_seq, b_seq, re2.padding_val_a, re2.padding_val_b)
    return model

def get_re2_char_model(hparams, vocab: NSVocabulary):
    from models.modules.stacked_encoder import StackedEncoder
    from models.modules.embedding_dropout import SeqEmbeddingDropoutWrapper
    from models.modules.word_char_embedding import WordCharEmbedding
    from models.matching.mha_encoder import MHAEncoder
    from models.matching.re2 import ChRE2
    from models.matching.re2_modules import Re2Block, Re2Prediction, Re2Conn, Re2Fusion, Re2Alignment, Re2Pooling
    from models.matching.re2_modules import NeoRe2Pooling, NeoFusion
    from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper

    word_emb_sz, hid_sz, dropout = hparams.emb_sz, hparams.hidden_size, hparams.dropout
    char_emb_sz, char_hid_sz = hparams.char_emb_sz, hparams.char_hid_sz
    d_dropout = hparams.discrete_dropout if hasattr(hparams, "discrete_dropout") else 0.
    i_dropout = hparams.dropout if hasattr(hparams, "dropout") else 0.

    embedding_a = nn.Embedding(vocab.get_vocab_size('nl'), word_emb_sz)
    embedding_char_a = nn.Embedding(vocab.get_vocab_size('nlch'), char_emb_sz)
    embedding_b = nn.Embedding(vocab.get_vocab_size('lf'), word_emb_sz)
    embedding_char_b = nn.Embedding(vocab.get_vocab_size('lfch'), char_emb_sz)

    embedding_a = SeqEmbeddingDropoutWrapper(embedding_a, d_dropout, i_dropout)
    embedding_char_a = SeqEmbeddingDropoutWrapper(embedding_char_a, d_dropout, i_dropout)
    embedding_b = SeqEmbeddingDropoutWrapper(embedding_b, d_dropout, i_dropout)
    embedding_char_b = SeqEmbeddingDropoutWrapper(embedding_char_b, d_dropout, i_dropout)


    wc_emb_a = WordCharEmbedding(embedding_a, embedding_char_a,
                                 PytorchSeq2SeqWrapper(
                                     nn.LSTM(char_emb_sz, char_emb_sz, batch_first=True),
                                 ))
    wc_emb_b = WordCharEmbedding(embedding_b, embedding_char_b,
                                 PytorchSeq2SeqWrapper(
                                     nn.LSTM(char_emb_sz, char_emb_sz, batch_first=True),
                                 ))

    emb_sz = word_emb_sz + char_emb_sz
    conn: Re2Conn = Re2Conn(hparams.connection, emb_sz, hid_sz)
    conn_out_sz = conn.get_output_size()

    if hasattr(hparams, "pooling") and hparams.pooling == "neo":
        # neo fusion outputs 3times larger embedding, contains max, mean, and std pooling
        pooling = NeoRe2Pooling()
        pred_inp = hid_sz * 3
    else:
        # common fusion uses only max pooling
        pooling = Re2Pooling()
        pred_inp = hid_sz

    pred = Re2Prediction(hparams.prediction, inp_sz=pred_inp, hid_sz=hid_sz,
                         num_classes=hparams.num_classes, dropout=dropout, activation=nn.SELU())

    def _encoder(inp_sz):
        if hasattr(hparams, "encoder") and hparams.encoder == "lstm":
            return PytorchSeq2SeqWrapper(
                torch.nn.LSTM(inp_sz, hid_sz, hparams.num_stacked_encoder,
                              batch_first=True, bidirectional=False,
                              dropout=dropout if hparams.num_stacked_encoder > 1 else 0.)
            )
        elif hasattr(hparams, "encoder") and hparams.encoder == "bilstm":
            return PytorchSeq2SeqWrapper(
                torch.nn.LSTM(inp_sz, hid_sz // 2, hparams.num_stacked_encoder,
                              batch_first=True, bidirectional=True,
                              dropout=dropout if hparams.num_stacked_encoder > 1 else 0.)
            )
        else:
            return StackedEncoder([
                MHAEncoder(inp_sz if j == 0 else hid_sz, hid_sz, hparams.num_heads, dropout)
                for j in range(hparams.num_stacked_encoder)
            ], inp_sz, hid_sz, dropout, output_every_layer=False)

    fusion_cls = NeoFusion if hasattr(hparams, "fusion") and hparams.fusion == "neo" else Re2Fusion

    enc_inp_sz = lambda i: emb_sz if i == 0 else conn_out_sz
    blocks = torch.nn.ModuleList([
        Re2Block(
            _encoder(enc_inp_sz(i)),
            _encoder(enc_inp_sz(i)),
            fusion_cls(hid_sz + enc_inp_sz(i), hid_sz, hparams.fusion == "full", dropout),
            fusion_cls(hid_sz + enc_inp_sz(i), hid_sz, hparams.fusion == "full", dropout),
            Re2Alignment(hid_sz + enc_inp_sz(i), hid_sz, hparams.alignment, nn.SELU()),
            dropout=dropout,
        )
        for i in range(hparams.num_stacked_block)
    ])

    model = ChRE2(vocab.get_token_index(PADDING_TOKEN, 'nlch'),
                  vocab.get_token_index(PADDING_TOKEN, 'lfch'),
                  wc_emb_a, wc_emb_b, blocks, pooling, pooling, conn, pred,
                  vocab.get_token_index(PADDING_TOKEN, 'nl'),
                  vocab.get_token_index(PADDING_TOKEN, 'lf'))
    return model

def get_char_giant(hparams, vocab: NSVocabulary):
    from models.matching.char_giant_ranker import CharGiantRanker
    from models.matching.re2 import ChRE2
    from models.matching.seq2seq_modeling import Seq2SeqModeling
    from models.matching.seq_modeling import SeqModeling, PytorchSeq2SeqWrapper
    from allennlp.modules.matrix_attention import BilinearMatrixAttention

    re2: ChRE2 = get_re2_char_model(hparams, vocab)
    emb_sz, hid_sz, dropout = hparams.emb_sz, hparams.hidden_size, hparams.dropout
    RNNWrapper = PytorchSeq2SeqWrapper
    get_rnn = lambda stateful: RNNWrapper(nn.LSTM(emb_sz, hid_sz, num_layers=hparams.num_stacked_encoder,
                                                  batch_first=True, dropout=hparams.dropout), stateful=stateful)

    a2b = Seq2SeqModeling(a_embedding=re2.a_emb.word_emb,
                          b_embedding=re2.b_emb.word_emb,
                          encoder=get_rnn(stateful=True),
                          decoder=get_rnn(stateful=False),
                          a_padding=re2.padding_val_a,
                          b_padding=re2.padding_val_b,
                          prediction=nn.Linear(hid_sz * 2, vocab.get_vocab_size('lf')),
                          attention=BilinearMatrixAttention(hid_sz, hid_sz),
                          )

    b2a = Seq2SeqModeling(a_embedding=re2.b_emb.word_emb,
                          b_embedding=re2.a_emb.word_emb,
                          encoder=get_rnn(stateful=True),
                          decoder=get_rnn(stateful=False),
                          a_padding=re2.padding_val_b,
                          b_padding=re2.padding_val_a,
                          prediction=nn.Linear(hid_sz * 2, vocab.get_vocab_size('nl')),
                          attention=BilinearMatrixAttention(hid_sz, hid_sz),
                          )

    a_seq = SeqModeling(embedding=re2.a_emb.word_emb,
                        encoder=get_rnn(stateful=False),
                        padding=re2.padding_val_a,
                        prediction=nn.Linear(hid_sz * 2, vocab.get_vocab_size('nl')),
                        attention=BilinearMatrixAttention(hid_sz, hid_sz),
                        )

    b_seq = SeqModeling(embedding=re2.b_emb.word_emb,
                        encoder=get_rnn(stateful=False),
                        padding=re2.padding_val_b,
                        prediction=nn.Linear(hid_sz * 2, vocab.get_vocab_size('lf')),
                        attention=BilinearMatrixAttention(hid_sz, hid_sz),
                        )

    model = CharGiantRanker(re2.a_emb.word_emb, re2.b_emb.word_emb,
                            re2, a2b, b2a, a_seq, b_seq,
                            re2.padding_val_a, re2.padding_val_b, re2.padding_char_a, re2.padding_char_b)
    return model

