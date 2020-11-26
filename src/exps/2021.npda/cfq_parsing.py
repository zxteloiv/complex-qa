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

def main(args):
    import datasets.cfq
    cfq = datasets.cfq.complete_cfq()
    print(len(cfq))

    pool = multiprocessing.Pool(args.workers, initializer=prepare, initargs=(args.grammar, args.entry))

    extended_dataset = list(pool.starmap(parse, enumerate(cfq), chunksize=args.chunk))
    import pickle
    pickle.dump(extended_dataset, open(args.output, 'wb'))

def prepare(grammar_filename, start):
    global parser
    parser = lark.Lark(open(grammar_filename), start=start)

def parse(i, data):
    global parser
    tree = {'sparql_tree': str(parser.parse(data['sparql'])),
            'sparqlPattern_tree': str(parser.parse(data['sparqlPattern'])),
            'sparqlPatternModEntities_tree': str(parser.parse(data['sparqlPatternModEntities']))}
    data.update(tree)
    data['raw_id'] = i
    return data

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--workers', '-w', type=int, default=4)
    import os.path
    default_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
                                                'statics', 'grammar', 'sparql_pattern.bnf.lark'))
    p.add_argument('--grammar', '-G', type=str, default=default_path)
    p.add_argument('--entry', '-e', type=str, default='queryunit')
    p.add_argument('--chunk', '-c', type=int, default=100)
    p.add_argument('--output', '-o', type=str, default="parsed_cfq.pkl")
    args = p.parse_args()
    main(args)



