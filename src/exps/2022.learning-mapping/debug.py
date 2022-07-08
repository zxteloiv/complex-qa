import sys
import os.path
from trialbot.training import Registry
from trialbot.utils.root_finder import find_root
sys.path.insert(0, find_root('.SRC'))


def foo():
    from trialbot.data import NSVocabulary
    from datasets.squall import install_squall_datasets
    import datasets.squall_translator
    install_squall_datasets()
    train, dev, test = Registry.get_dataset('squall0')
    print('lengths:', len(train), len(dev), len(test))
    from trialbot.data.translator import FieldAwareTranslator
    translator: FieldAwareTranslator = Registry.get_translator('squall-base')
    from utils.vocab_builder import get_ns_counter

    if os.path.exists('temp-vocab'):
        vocab = NSVocabulary.from_files('temp-vocab')
    else:
        vocab = NSVocabulary(get_ns_counter(train, translator))
        vocab.save_to_files('temp-vocab')
    translator.index_with_vocab(vocab)
    print(vocab)

    from datasets.squall_translator import SquallAllInOneField
    field: SquallAllInOneField = translator.fields[0]

    for i, example in enumerate(train):
        ex = translator.to_tensor(example)
        inspect_ex(example, ex, field)
        batch = translator.batch_tensor([ex])
        break


def inspect_ex(example, ex, field):
    from datasets.squall_translator import SquallAllInOneField
    field: SquallAllInOneField
    from transformers import BertTokenizer
    berttok: BertTokenizer = field._tokenizer
    nl, sql = example['nl'], example['sql']
    print(f'nl_toks: {" ".join(f"({i}){x}" for i, x in enumerate(nl))}')
    print(f'sql_toks: {" ".join(f"({i}){x[1]}" for i, x in enumerate(sql))}')
    print(f'columns: {" ".join(f"({i}){x[0]}" for i, x in enumerate(example["columns"]))}')
    print('aligns:')
    for (srcspan, tgtspan) in example['align']:
        srcwords = ' '.join(nl[i] for i in srcspan)
        tgtwords = ' '.join(str(sql[i][1]) for i in tgtspan)
        print(f"    {srcwords} <-->  {tgtwords}")
    print('-----' * 20)

    print("rev_toks:")
    plm_type_ids = ex['src_plm_type_ids']
    x_ids = ex['src_ids']
    src_type = ex['src_types']
    rev_id = {0: 'pad', 1: 'special', 2: 'word', 3: 'word_pivot', 4: 'column', 5: 'col_pivot'}
    rev_tid = {0: 'pad', 1: 'key', 2: 'col', 3: 'strval', 4: 'numval'}

    def get_sql_i(i):
        t = ex['tgt_type'][i]
        if t == 0:
            out = '<pad>'
        elif t == 1:
            out = field.vocab.get_token_from_index(ex['tgt_keyword'][i], 'keyword')
        elif t == 2:
            col_type = field.vocab.get_token_from_index(ex['tgt_col_type'][i], 'col_type')
            col_id = ex['tgt_col_id'][i]
            col = berttok.convert_ids_to_tokens(x_ids[col_id])
            out = f"col'{col}'({col_id}-{col_type})"
        elif t == 3:
            begin = ex['tgt_literal_begin'][i]
            end = ex['tgt_literal_end'][i]
            out = berttok.convert_ids_to_tokens([x_ids[begin], x_ids[end]])
        elif t == 4:
            begin = ex['tgt_literal_begin'][i]
            out = berttok.convert_ids_to_tokens([x_ids[begin]])
        return f"({s}-{rev_tid[t]}) {out}"

    for i, (tok, st, pt) in enumerate(zip(berttok.convert_ids_to_tokens(x_ids), src_type, plm_type_ids)):
        st = rev_id[st]
        print(f"({i})\t{tok} ({st}, {pt})")
    print('-----' * 20)
    for w, s in zip(ex['align_ws_word'], ex['align_ws_sql']):
        print(f'({w}) {berttok.convert_ids_to_tokens([x_ids[w]])} <----> {get_sql_i(s)}')

    for w, c in zip(ex['align_wc_word'], ex['align_wc_col']):
        print(f'({w}) {berttok.convert_ids_to_tokens(x_ids[w])} <--> ({c}) {berttok.convert_ids_to_tokens(x_ids[c])}')

    for s, c in zip(ex['align_sc_sql'], ex['align_sc_col']):
        print(f'{get_sql_i(s)} <----> ({c}) {berttok.convert_ids_to_tokens([x_ids[c]])}')

    print('-----' * 20)
    print(f'type: {" ".join(f"({i}, {rev_tid[x]})" for i, x in enumerate(ex["tgt_type"]))}')
    for i in range(len(ex['tgt_type'])):
        print(get_sql_i(i))


if __name__ == '__main__':
    foo()