__doc__ = """
Script to parse the cfq dataset into trees in Lark.
The grammar used is compiled from ebnf_compiler.
"""

import sys
import os.path
sys.path.insert(0, os.path.join('..', '..'))
import lark
import multiprocessing

parser: lark.Lark = None

def get_cfq(infile):
    import json
    for l in open(infile):
        yield json.loads(l.rstrip())

def main(args):
    cfq = get_cfq(args.input)
    import pickle
    if args.workers > 1:
        pool = multiprocessing.Pool(args.workers, initializer=prepare, initargs=(args.grammar, args.entry))
        extended_dataset = list(pool.starmap(parse, enumerate(cfq), chunksize=args.chunk))
    else:
        extended_dataset = list()
        prepare(args.grammar, args.entry)
        from datetime import datetime as dt
        for i, data in enumerate(cfq):
            data = parse(i, data)
            if i % 100 == 0:
                print(i, "...", dt.now().strftime('%H:%M:%S'), flush=True)
            extended_dataset.append(data)

    pickle.dump(extended_dataset, open(args.output, 'wb'))

def prepare(grammar_filename, start):
    global parser
    parser = lark.Lark(open(grammar_filename), start=start, keep_all_tokens=True)

def parse(i, data):
    global parser
    tree = {'sparql_tree': parser.parse(data['sparql']),
            'sparqlPattern_tree': parser.parse(data['sparqlPattern']),
            'sparqlPatternModEntities_tree': parser.parse(data['sparqlPatternModEntities'])}
    data.update(tree)
    data['raw_id'] = i
    return data

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--workers', '-w', type=int, default=4)
    import os.path
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    default_path = os.path.join(ROOT, 'src', 'statics', 'grammar', 'sparql_pattern.bnf.lark')
    default_input = os.path.join(ROOT, 'data', 'cfq', 'dataset_slim.jsonl')
    p.add_argument('--grammar', '-G', type=str, default=default_path)
    p.add_argument('--entry', '-e', type=str, default='queryunit')
    p.add_argument('--chunk', '-c', type=int, default=100)
    p.add_argument('--input', '-i', type=str, default=default_input)
    p.add_argument('--output', '-o', type=str, default="parsed_cfq.pkl")
    args = p.parse_args()
    main(args)



