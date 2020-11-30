import redis
import pickle
import logging
import lark
from datetime import datetime as dt

def timer():
    return dt.now().strftime('%H:%M:%S')

def main(args):
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    pool = redis.ConnectionPool(host=args.host, port=args.port, db=args.db)
    r = redis.Redis(connection_pool=pool)
    logger.info(f'Cache initialized as {args.host}:{args.port} for the db {args.db} ... {timer()}')

    logger.info(f'Reading file {args.file} ... {timer()}')
    data = pickle.load(open(args.file, 'rb'))
    logger.info(f'Start to warm up cache ... {timer()}')
    r.set(args.prefix + 'len', len(data))
    logger.info(f'  write length {len(data)} ... {timer()}')
    for i, x in enumerate(data):
        errno = r.set(args.prefix + str(i), pickle.dumps(x))
        if i % 1000 == 0:
            logger.info(f'  write {i}th example ... {timer()}')
    logger.info(f'Finished cache warm-up ... {timer()}')

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    import os.path
    DATAPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data'))
    default_path = os.path.join(DATAPATH, 'cfq', 'parsed_cfq.pkl')
    p.add_argument('--file', type=str, default=default_path)
    p.add_argument('--host', type=str, default='localhost')
    p.add_argument('--port', type=int, default=6379)
    p.add_argument('--db', type=int, default=0)
    p.add_argument('--prefix', type=str, default='cfq_parse_')
    args = p.parse_args()
    main(args)

