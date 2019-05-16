from .geoquery import GeoQueryDatasetReader
from .tab_sep_seqs import TabSepDatasetReader, TabSepJiebaCutReader, TabSepSharedVocabReader, TabSepCharReader
from .spider import SpiderDatasetReader

DATA_READERS = dict()
DATA_READERS['geoquery'] = GeoQueryDatasetReader
DATA_READERS['tab_sep'] = TabSepDatasetReader
DATA_READERS['tab_sep_shared'] = TabSepSharedVocabReader
DATA_READERS['tab_sep_jieba'] = TabSepJiebaCutReader
DATA_READERS['tab_sep_char'] = TabSepCharReader
DATA_READERS['spider'] = SpiderDatasetReader
