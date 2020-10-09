from utils.vocab_builder import get_ns_counter
from datasets.complex_web_q_translator import CompWebQTranslator
from trialbot.data.ns_vocabulary import NSVocabulary
from datasets.complex_web_q import complex_web_q

def main(output_path: str):
    train, _, _ = complex_web_q()
    translator = CompWebQTranslator(max_lf_len=10000, max_nl_len=10000)
    counter = get_ns_counter(train, translator)
    vocab = NSVocabulary(counter,
                         min_count=dict((ns, 5) for ns in translator.ns))
    print(f"saving vocab to {output_path}")
    vocab.save_to_files(output_path)
    print(vocab)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main('compwebq_vocab_5')