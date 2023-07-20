from collections.abc import Callable
from transformers import AutoTokenizer


def get_llm_wrapper_split_fn(llm_path: str) -> Callable[[str | list[str]], list[str]]:
    tok = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)

    def split_fn(sent: str | list[str]) -> list[str]:
        if isinstance(sent, list):
            sent = ' '.join(sent)
        ids: list[int] = tok(sent).input_ids
        return [str(x) for x in ids]

    return split_fn


