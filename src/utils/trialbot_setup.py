from typing import Mapping, Optional

def install_dataset_into_registry(datasets: dict):
    from trialbot.training import Registry
    Registry._datasets.update(datasets)

def setup(**default_args):
    """
    parse the setup and set default function as a boilerplate.
    :param default_args:
    :return:
    """
    from trialbot.training import TrialBot
    import sys, logging
    args = sys.argv[1:]
    for argname, argval in default_args.items():
        if f'--{argname}' not in args:
            args += [f'--{argname}', str(argval)]

    parser = TrialBot.get_default_parser()
    parser.add_argument('--dev', action='store_true', help="use dev data for testing mode, only works with --test opt")
    args = parser.parse_args(args)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.INFO)

    if hasattr(args, "seed") and args.seed:
        from utils.fix_seed import fix_seed
        logging.info(f"set seed={args.seed}")
        fix_seed(args.seed)

    return args

if __name__ == '__main__':
    print(setup(seed=2020))

