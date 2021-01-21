import sys
sys.path.insert(0, '../../..')
import datasets.django_rank
from collections import defaultdict

_, dev, _ = datasets.django_rank.django_five()

def process_hyp(e):
    hyp_rank, src, tgt, hyp, is_correct = list(map(e.get, ("hyp_rank", "src", "tgt", "hyp", "is_correct")))
    if 'def ' in tgt:
        return def_pat(hyp), def_pat(tgt), hyp_rank, is_correct

def def_pat(hyp):
    import re
    pat = re.sub('[a-zA-Z_0-9]+(=[a-zA-Z0-9_]+)?', 'a', hyp).replace('\n', ' ')
    pat = re.sub('  *', ' ', pat)
    return pat

total = len(dev)
dev_features = list(filter(None, [process_hyp(e) for e in dev]))

pat_rank = defaultdict(list)
for (pat_hyp, pat_tgt, hyp_rank, is_correct) in dev_features:
    if is_correct:
        pat_rank[pat_tgt].append(int(hyp_rank))

for k, v in pat_rank.items():
    if len(v) > 2:
        print(f"{k}: {sum(v) / len(v)} of {len(v)} in {len(dev_features)}")
