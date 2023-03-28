# a utility to analyze the logs output from the
from typing import List
import json
import numpy as np
import sys
from trialbot.utils.root_finder import find_root
sys.path.insert(0, find_root('.SRC'))


def process_tranx_log(filename: str):
    if 'log.cuda' in filename or 'log.' not in filename:
        return

    from utils.text_tool import scan, grep, split

    ds_name, split_tag, model_name, seed = parse_filename(filename)

    data: List[str] = [l.strip() for l in open(filename) if len(l.strip()) > 0]
    if len(grep(data, 'completed')) == 0:
        print('invalid file:', filename)
        return

    groups = split(scan(data, 'Epoch started'), '-------*')
    num_epoch = len(groups)
    train_err, dev_err, test_err = [], [], []
    for g in groups[:]:
        train_metrics = grep(grep(g, ':{.*}|Training Metrics'), '{.*}', only_matched_text=True)[-1]
        dev_metrics = grep(grep(g, 'Evaluation Metrics'), '{.*}', only_matched_text=True)[-1]
        test_metrics = grep(grep(g, 'Testing Metrics'), '{.*}', only_matched_text=True)[-1]
        train_err.append(json.loads(train_metrics)['ERR'])
        dev_err.append(json.loads(dev_metrics)['ERR'])
        test_err.append(json.loads(test_metrics)['ERR'])

    train_min_err = np.min(train_err).item()
    train_min_where = np.argmin(train_err).item()
    dev_min_err = np.min(dev_err).item()
    dev_min_where = np.argmin(dev_err).item()
    test_min_err = np.min(test_err).item()
    test_min_where = np.argmin(test_err).item()
    test_min_rel = test_err[dev_min_where]

    print(filename, ds_name, split_tag, model_name, seed, num_epoch,
          train_min_err, train_min_where, dev_min_err, dev_min_where,
          test_min_err, test_min_where, test_min_rel,
          sep='\t')


def parse_filename(filename: str):
    import re
    m = re.search(r'log\.(atis|scholar|geo|advising|raw_qa_all)_(iid|cg)\.([a-z_\d]+)\.s(\d+)', filename)
    ds = m.group(1)
    split = m.group(2)
    model = m.group(3)
    seed = m.group(4)
    return ds, split, model, seed


def main():
    process_tranx_log(sys.argv[1])


if __name__ == '__main__':
    main()

