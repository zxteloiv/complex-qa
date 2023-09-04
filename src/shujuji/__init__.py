# if you ever laughed at this directory name
# try instead to criticize HuggingFace
# who had robbed the "datasets" name at PyPI.

PREF_CONFS = {
    'cogs': ('nl', 'lf'),
    'geo': ('sent', 'sql'),
    'ati': ('sent', 'sql'),
    'sch': ('sent', 'sql'),
    'adv': ('sent', 'sql'),
    'smc': ('utterance', 'plan'),
    'ccfq': ('source', 'target'),
    'cofe': ('context', 'ground_truth'),
    'agsa': ('sent', 'sql'),
}


def get_field_names_by_prefix(ds_name: str) -> tuple[str, str]:
    for k, v in PREF_CONFS.items():
        if ds_name.startswith(k):
            return v
    raise ValueError(f'Unknown prefix of the dataset {ds_name}.')


def install_semantic_parsing_datasets(reg: dict = None):
    if reg is None:
        from trialbot.training import Registry
        reg = Registry._datasets

    from . import (
        cogs,
        compact_cfq as ccfq,
        smcalflow_cs as smc,
        cg_bundle as agsa,
        cofe,
    )

    smc.install(reg)
    ccfq.install_dataset(reg)
    cogs.install_dataset(reg)
    agsa.install_raw_qa_datasets(reg)
    agsa.install_parsed_qa_datasets(reg)
    agsa.install_cross_domain_raw_qa_datasets(reg)
    agsa.install_cross_domain_parsed_qa_datasets(reg)
    cofe.install_datasets(reg)


