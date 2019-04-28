from .geoquery import GeoQueryDatasetReader
from .tab_sep_seqs import TabSepDatasetReader, TabSepJiebaCutReader

DATA_READERS = dict()
DATA_READERS['geoquery'] = GeoQueryDatasetReader
DATA_READERS['tab_sep'] = TabSepDatasetReader
DATA_READERS['tab_sep_jieba'] = TabSepJiebaCutReader
