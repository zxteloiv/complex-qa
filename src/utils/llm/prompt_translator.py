import logging
from typing import Mapping, Generator, Tuple, List
from trialbot.data.translator import Translator, NullableTensor, FieldAwareTranslator
from .prompt import build_ctx_with_exemplars, build_prompt
from ..ir_client import VolatileBM25Index


class PromptTranslator(Translator):
    def batch_tensor(self, tensors: List[Mapping[str, NullableTensor]]):
        tensors = list(filter(lambda x: all(v is not None for v in x.values()), tensors))
        batch_dict = FieldAwareTranslator.list_of_dict_to_dict_of_list(tensors)
        return batch_dict

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        yield from []

    def to_tensor(self, example) -> Mapping[str, NullableTensor]:
        src, tgt = map(example.get, (self.src_field, self.tgt_field))

        # use empty string or empty list for absent fields
        # because None has been reserved for illegal values
        if self.prompt_mode == 'zero_shot':
            return {'src': src, 'tgt': tgt, 'context': [], 'prompt': src}

        exemplars: List[dict] = self.search_index(src)
        if self.prompt_mode == 'history':
            ctx = build_ctx_with_exemplars(
                exemplars, self.src_field, self.tgt_field,
                use_ast='none',  # never use syntactic parse in the context
            )
            return {'src': src, 'tgt': tgt, 'context': ctx, 'prompt': ''}

        elif self.prompt_mode == 'prompt':
            ctx = build_ctx_with_exemplars(exemplars, self.src_field, self.tgt_field, 'none', None)
            prompt = build_prompt(src, ctx)
            return {'src': src, 'tgt': tgt, 'context': ctx, 'prompt': prompt}

        elif self.prompt_mode == 'syn_prompt':
            ctx = build_ctx_with_exemplars(exemplars, self.src_field, self.tgt_field, 'as_brackets')
            prompt = build_prompt(src, ctx)
            return {'src': src, 'tgt': tgt, 'context': ctx, 'prompt': prompt}

        else:
            raise ValueError(f'Not supported prompt mode {self.prompt_mode}.')

    def __init__(self,
                 src_field: str,
                 tgt_field: str,
                 prompt_mode: str = 'zero_shot',  # 'zero_shot', 'history', 'prompt', 'syn_prompt'
                 num_exemplars: int = 10,
                 ):
        super().__init__()
        self.idx_conn = None
        self.src_field = src_field
        self.tgt_field = tgt_field
        self.indexing_dataset = None
        self.prompt_mode = prompt_mode
        self.num_exemplars = num_exemplars

    def load_index(self, train_set):
        if self.idx_conn is not None:
            logging.getLogger(self.__class__.__name__).info('INDEX ALREADY BUILT. Skip loading.')
            return

        self.idx_conn = VolatileBM25Index.from_data_list(
            keys=[x.get(self.src_field) for x in train_set],
            payloads=list(range(len(train_set))),
            default_search_limit=self.num_exemplars,
        )
        self.indexing_dataset = train_set

    def search_index(self, key: str) -> List[dict]:
        return [self.indexing_dataset[i] for _, i in self.idx_conn.search_index(key)]


def install_translator(reg: dict = None):
    if reg is None:
        from trialbot.training import Registry
        reg = Registry._translators

    reg['prompt'] = PromptTranslator
