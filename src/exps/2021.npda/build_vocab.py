import os.path, sys
sys.path.insert(0, '../..')
from utils.vocab_builder import get_ns_counter
from trialbot.data import NSVocabulary

def main():
    build_cfq_pda_vocab(sys.argv[1])

def build_cfq_pda_vocab(name):
    from shujuji import cfq, cfq_translator, lark_translator
    grammar_ds, _, _ = cfq.sparql_pattern_grammar()
    cfq_ds, _, _ = cfq.cfq_mcd1()

    print("counting grammar tokens...")
    counter = get_ns_counter(grammar_ds, lark_translator.UnifiedLarkTranslator(cfq_translator.UNIFIED_TREE_NS))
    print("counting cfq tokens...")
    counter.update(get_ns_counter(cfq_ds, cfq_translator.CFQFlatDerivations()))

    vocab = NSVocabulary(counter)
    print("saving vocab...")
    vocab.save_to_files(name)

if __name__ == '__main__':
    main()