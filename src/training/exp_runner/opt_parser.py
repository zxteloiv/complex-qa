import argparse
import config
import data_adapter

def get_common_opt_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, help='manually set the seeds for torch')
    parser.add_argument('--device', type=int, default=-1, help="the gpu device number to override")
    parser.add_argument("--quiet", action="store_true", help="mute the log")
    parser.add_argument("--debug", action="store_true", help="print the debugging log")
    parser.add_argument('--memo', type=str, default="", help="used to remember some runtime configurations")
    parser.add_argument('--test', action="store_true", help='use testing mode')
    parser.add_argument('--hparamset', help="available hyper-parameters")
    parser.add_argument('--from-hparamset-dump', help='read hyperparameters from the dump file, to reproduce')
    parser.add_argument('--list-hparamset', action='store_true')
    parser.add_argument('--snapshot-dir', help="snapshot dir if continues")
    parser.add_argument('--dataset', choices=config.DATASETS.keys())
    parser.add_argument('--data-reader', choices=data_adapter.DATA_READERS.keys())

    return parser

def get_test_opt_parser() -> argparse.ArgumentParser:
    parser = get_common_opt_parser()
    return parser
