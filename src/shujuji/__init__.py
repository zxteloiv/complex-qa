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
}


def get_field_names_by_prefix(ds_name: str):
    from shujuji import PREF_CONFS
    for k, v in PREF_CONFS.items():
        if ds_name.startswith(k):
            return v
    return None


