from typing import Optional, List
from trialbot.data import NSVocabulary
import torch

def make_human_readable_text(output_tensor: torch.Tensor,
                             vocab: NSVocabulary,
                             ns: str,
                             stop_ids: Optional[list] = None
                             ) -> list:
    """Convert the predicted word ids into discrete tokens"""
    import numpy as np

    # output_tensor: (batch, max_length)
    if not isinstance(output_tensor, np.ndarray):
        output_tensor = output_tensor.detach().cpu().numpy()

    all_readable_tokens = []
    for token_ids in output_tensor:
        if token_ids.ndim > 1:
            token_ids = token_ids[0]

        token_ids = list(token_ids)
        for stop_tok in stop_ids:
            if stop_tok in token_ids:
                token_ids = token_ids[:token_ids.index(stop_tok)]

        tokens = [vocab.get_token_from_index(token_id, namespace=ns) for token_id in token_ids]
        all_readable_tokens.append(tokens)
    return all_readable_tokens
