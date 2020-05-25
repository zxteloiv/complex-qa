import io
import tokenize
from .atis_rank_translator import AtisRankTranslator, AtisRankChTranslator
from trialbot.training import Registry

import token as tk
from io import StringIO
from tokenize import generate_tokens


def tokenize_code(code, mode=None):
    token_stream = generate_tokens(StringIO(code).readline)
    tokens = []
    for toknum, tokval, (srow, scol), (erow, ecol), _ in token_stream:
        if toknum == tk.ENDMARKER:
            break

        if mode == 'decoder':
            if toknum == tk.STRING:
                quote = tokval[0]
                tokval = tokval[1:-1]
                tokens.append(quote)
                tokens.append(tokval)
                tokens.append(quote)
            elif toknum == tk.DEDENT:
                continue
            else:
                tokens.append(tokval)
        elif mode == 'canonicalize':
            if toknum == tk.STRING:
                tokens.append('_STR_')
            elif toknum == tk.DEDENT:
                continue
            else:
                tokens.append(tokval)
        else:
            tokens.append(tokval)

    return tokens

@Registry.translator('django_rank')
class DjangoRankTranslator(AtisRankTranslator):
    def __init__(self, max_len=50):
        super().__init__(max_len)

    def split_lf_seq(self, seq: str):
        g = tokenize_code(seq, mode="decoder")
        return list(tok for tok in g)

@Registry.translator('django_rank_char')
class DjangoRankChTranslator(AtisRankChTranslator):
    def __init__(self, max_len=50):
        super().__init__(max_len)

    def split_lf_seq(self, seq: str):
        g = tokenize_code(seq, mode="decoder")
        return list(tok for tok in g)


if __name__ == '__main__':
    print(tokenize_code('offset = self.getpos()()'))
