import logging
import sys, os.path as osp
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..', '..')))   # up to src
from utils.trialbot.setup import setup
import tranx_bin

def main():
    exp_args = [
        dict(dataset='atis_iid.handcrafted', seed=2021, hparamset='atis_common', device=0),
        # dict(dataset='atis_iid.handcrafted', seed=2022, hparamset='atis_common', device=0),
        # dict(dataset='atis_iid.handcrafted', seed=2023, hparamset='atis_common', device=0),
        # dict(dataset='atis_iid.handcrafted', seed=2024, hparamset='atis_common', device=0),
        # dict(dataset='atis_iid.handcrafted', seed=2025, hparamset='atis_common', device=0),
        dict(dataset='atis_cg.handcrafted', seed=2021, hparamset='atis_common', device=0),
        # dict(dataset='atis_cg.handcrafted', seed=2022, hparamset='atis_common', device=0),
        # dict(dataset='atis_cg.handcrafted', seed=2023, hparamset='atis_common', device=0),
        # dict(dataset='atis_cg.handcrafted', seed=2024, hparamset='atis_common', device=0),
        # dict(dataset='atis_cg.handcrafted', seed=2025, hparamset='atis_common', device=0),
        # dict(dataset='scholar_iid.handcrafted', seed=2021, hparamset='scholar_common', device=0),
        # dict(dataset='scholar_iid.handcrafted', seed=2022, hparamset='scholar_common', device=0),
        # dict(dataset='scholar_iid.handcrafted', seed=2023, hparamset='scholar_common', device=0),
        # dict(dataset='scholar_iid.handcrafted', seed=2024, hparamset='scholar_common', device=0),
        # dict(dataset='scholar_iid.handcrafted', seed=2025, hparamset='scholar_common', device=0),
        # dict(dataset='scholar_cg.handcrafted', seed=2021, hparamset='scholar_common', device=0),
        # dict(dataset='scholar_cg.handcrafted', seed=2022, hparamset='scholar_common', device=0),
        # dict(dataset='scholar_cg.handcrafted', seed=2023, hparamset='scholar_common', device=0),
        # dict(dataset='scholar_cg.handcrafted', seed=2024, hparamset='scholar_common', device=0),
        # dict(dataset='scholar_cg.handcrafted', seed=2025, hparamset='scholar_common', device=0),
    ]

    sys.argv[1:] = []
    for i, args in enumerate(exp_args):
        logfile_name = f'exp_{i}_{args["dataset"].split(".")[0]}_s{args["seed"]}.log'
        logging.basicConfig(filename=logfile_name, force=True)  # reset the handlers for each exp
        tranx_bin.run_exp(setup(translator='tranx_no_terminal', **args))


if __name__ == '__main__':
    main()
