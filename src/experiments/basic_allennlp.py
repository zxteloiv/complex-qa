"""
<h1>Welcome</h1>
<p>Welcome to AllenNLP! This tutorial will walk you through the basics of building and training an AllenNLP model.</p>
 {% include more-tutorials.html %}
 <p>Before we get started, make sure you have a clean Python 3.6 or 3.7 virtual environment, and then run the following command to install the AllenNLP library:</p>
 {% highlight bash %}
pip install allennlp
{% endhighlight %}
 <hr />
 <p>In this tutorial we'll implement a slightly enhanced version of the PyTorch
<a href = "https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#example-an-lstm-for-part-of-speech-tagging">LSTM for Part-of-Speech Tagging</a> tutorial,
adding some features that make it a slightly more realistic task (and that also showcase some of the benefits of AllenNLP):</p>
 <ol class="formatted">
  <li>We'll read our data from files. (The tutorial example uses data that's given as part of the Python code.)</li>
  <li>We'll use a separate validation dataset to check our performance. (The tutorial example trains and evaluates on the same dataset.)</li>
  <li>We'll use <a href="https://github.com/tqdm/tqdm" target="_blank">tqdm</a> to track the progress of our training.</li>
  <li>We'll implement <a href="https://en.wikipedia.org/wiki/Early_stopping" target="_blank">early stopping</a> based on the loss on the validation dataset.</li>
  <li>We'll track accuracy on both the training and validation sets as we train the model.</li>
</ol>
 <p>(In addition to what's highlighted in this tutorial, AllenNLP provides many other "for free" features.)
 <hr />
 <h2>The Problem</h2>
 <p>Given a sentence (e.g. <code>"The dog ate the apple"</code>) we want to predict part-of-speech tags for each word<br />(e.g <code>["DET", "NN", "V", "DET", "NN"]</code>).</p>
 <p>As in the PyTorch tutorial, we'll embed each word in a low-dimensional space, pass them through an LSTM to get a sequence of encodings, and use a feedforward layer to transform those into a sequence of logits (corresponding to the possible part-of-speech tags).</p>
 <p>Below is the annotated code for accomplishing this. You can start reading the annotations from the top, or just look through the code and look to the annotations when you need more explanation.</p>
 <!-- Annotated Code -->
"""

from typing import Iterator, List, Dict


import torch
import torch.optim as optim
import numpy as np


from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField


from allennlp.data.dataset_readers import DatasetReader


from allennlp.common.file_utils import cached_path


from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token


from allennlp.data.vocabulary import Vocabulary


from allennlp.models import Model


from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits


from allennlp.training.metrics import CategoricalAccuracy


from allennlp.data.iterators import BucketIterator


from allennlp.training.trainer import Trainer


from allennlp.predictors import SentenceTaggerPredictor

torch.manual_seed(1)


class PosDatasetReader(DatasetReader):
    """
    DatasetReader for PoS tagging data, one sentence per line, like

        The###DET dog###NN ate###V the###DET apple###NN
    """

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}


    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}

        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
            fields["labels"] = label_field

        return Instance(fields)


    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                pairs = line.strip().split()
                sentence, tags = zip(*(pair.split("###") for pair in pairs))
                yield self.text_to_instance([Token(word) for word in sentence], tags)


class LstmTagger(Model):

    def __init__(self,

                 word_embeddings: TextFieldEmbedder,

                 encoder: Seq2SeqEncoder,

                 vocab: Vocabulary) -> None:

        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder

        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))

        self.accuracy = CategoricalAccuracy()


    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> torch.Tensor:

        mask = get_text_field_mask(sentence)

        embeddings = self.word_embeddings(sentence)

        encoder_out = self.encoder(embeddings, mask)

        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}


        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)

        return output


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}



reader = PosDatasetReader()

train_dataset = reader.read(cached_path(
    'https://raw.githubusercontent.com/allenai/allennlp'
    '/master/tutorials/tagger/training.txt'))
validation_dataset = reader.read(cached_path(
    'https://raw.githubusercontent.com/allenai/allennlp'
    '/master/tutorials/tagger/validation.txt'))


vocab = Vocabulary.from_instances(train_dataset + validation_dataset)


EMBEDDING_DIM = 6
HIDDEN_DIM = 6


token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))


model = LstmTagger(word_embeddings, lstm, vocab)


optimizer = optim.SGD(model.parameters(), lr=0.1)


iterator = BucketIterator(batch_size=2, sorting_keys=[("sentence", "num_tokens")])

iterator.index_with(vocab)


trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  patience=10,
                  num_epochs=1000)


trainer.train()


predictor = SentenceTaggerPredictor(model, dataset_reader=reader)

tag_logits = predictor.predict("The dog ate the apple")['tag_logits']

tag_ids = np.argmax(tag_logits, axis=-1)

print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])
