import logging
import sys
from trialbot.utils.fix_seed import fix_seed
from trialbot.training import TrialBot
import argparse


def augment_parser(parser: argparse.ArgumentParser):
    return parser


def setup(**default_args):
    """
    parse the setup and set default function as a boilerplate.
    :param default_args: set args if not found from commandline (sys.argv)
    :return:
    """
    argv = sys.argv[1:].copy()
    for argname, argval in default_args.items():
        if f'--{argname}' not in argv:
            argv += [f'--{argname}', str(argval)]

    parser = augment_parser(TrialBot.get_default_parser())
    args = parser.parse_args(argv)
    handle_common_args(args)
    return args


def setup_null_argv(**kwargs):
    argv = []
    for k, v in kwargs.items():
        argv.append(f'--{k}')
        argv.append(f'{v}')

    parser = augment_parser(TrialBot.get_default_parser())
    args = parser.parse_args(argv)
    handle_common_args(args)
    return args


def handle_common_args(args):
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.INFO)

    if hasattr(args, "seed") and args.seed:
        logging.info(f"set seed={args.seed}")
        fix_seed(args.seed)


if __name__ == '__main__':
    print(setup(seed=2020))

