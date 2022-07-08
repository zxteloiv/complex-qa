import logging
import sys
from trialbot.utils.fix_seed import fix_seed
from trialbot.training import TrialBot
import argparse


def augment_parser(parser: argparse.ArgumentParser):
    parser.add_argument('--log-to-file', '-L', action='store_true')
    return parser


def setup(**default_args):
    """
    parse the setup and set default function as a boilerplate.
    :param default_args: set args if not found from commandline (sys.argv)
    :return:
    """
    argv = sys.argv[1:].copy()
    defaults = []
    for argname, argval in default_args.items():
        defaults += [f'--{argname}', str(argval)]

    parser = augment_parser(TrialBot.get_default_parser())
    args = parser.parse_args(defaults + argv)
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
    if args.log_to_file:
        print('log to file set')
        def name_norm(s: str):
            import re
            return re.sub('[^a-zA-Z0-9_]', '_', s)

        hp = name_norm(args.hparamset)
        ds = name_norm(args.dataset)
        logfile_name = f'log.{ds}.{hp}.s{args.seed}'
        logging.basicConfig(filename=logfile_name, force=True)  # reset the handlers for each exp

    if isinstance(args.device, int) and args.device >= 0:
        # by setting this environ, the args.device will be mapped as device 0, and the move_to_device will fail.
        # import os
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        import torch
        torch.cuda.set_device(args.device)

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

