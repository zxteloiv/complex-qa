# Construct similarity matrix by BERT and natural language
# For every dataset

from sklearn.neighbors import NearestNeighbors
import torch
import numpy as np
import transformers
import pickle
import logging
logging.basicConfig(level=logging.INFO)

class ConstructBertSimilarity:
    def __init__(self, key="bert-base-cased", device=-1):

        self.tokenizer = transformers.BertTokenizer.from_pretrained(key)
        self.model = transformers.BertModel.from_pretrained(key)
        if device >= 0:
            self.model = self.model.cuda(device)

        self.device = device
        self.logger = logging.getLogger(__name__)

    def get_dataset_embedding(self, dataset):
        example_embeddings = []
        eids = []
        for example in dataset:
            eid = example['ex_id']
            sent = example['src']
            input_ids = torch.tensor([self.tokenizer.encode(sent)])
            if self.device >= 0:
                input_ids = input_ids.cuda(torch.device('cuda:' + str(self.device)))

            with torch.no_grad():
                output = self.model(input_ids)
                sent_emb = output[0][0, 1:-1].mean(0)
                sent_emb = sent_emb.cpu().numpy()
                example_embeddings.append(sent_emb)
                eids.append(eid)

        return eids, np.stack(example_embeddings)

    def build(self, num_nearest: int = 30, get_dataset_fn = None):
        train_set, dev_set, test_set = get_dataset_fn()
        log = self.logger

        log.info("To obtaion embedding matrix for the whole training dataset...")
        eids, embs = self.get_dataset_embedding(train_set)
        eid_to_rowid = dict(zip(eids, range(len(eids))))
        rowid_to_eid = dict(enumerate(eids))

        log.info("Building neighbor index...")
        nbrs = NearestNeighbors(n_neighbors=num_nearest, n_jobs=4).fit(embs)

        log.info("Querying the index with the training dataset")
        dist_train, nbridx_train = nbrs.kneighbors(embs)

        log.info("Querying the index with the testing dataset")
        test_eids, test_embs = self.get_dataset_embedding(test_set)
        dist_test, nbridx_test = nbrs.kneighbors(test_embs)

        log.info("Querying the index with the dev dataset")
        dev_eids, dev_embs = self.get_dataset_embedding(dev_set)
        dist_dev, nbridx_dev = nbrs.kneighbors(dev_embs)

        return nbridx_train, nbridx_dev, nbridx_test


if __name__ == '__main__':
    import sys
    sys.path = ['..'] + sys.path
    import os.path
    from utils.root_finder import find_root
    sim_dir = os.path.join(find_root(), 'data', '_similarity_index')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--dataset', type=str, choices=['atis', 'django'])
    parser.add_argument('--output-dir', type=str, default=sim_dir)
    args = parser.parse_args()

    cons = ConstructBertSimilarity(device=args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset == "atis":
        bert_sim_path = os.path.join(args.output_dir, 'atis_bert_nl.bin')
        from datasets.atis_rank import atis_pure_none
        nbridx = cons.build(30, atis_pure_none)

    elif args.dataset == "django":
        bert_sim_path = os.path.join(args.output_dir, 'django_bert_nl.bin')
        from datasets.django_rank import django_pure_none
        nbridx = cons.build(30, django_pure_none)

    else:
        raise Exception

    cons.logger.info('Dump the similarity lookup table...')
    pickle.dump(nbridx, open(bert_sim_path, 'wb'))



