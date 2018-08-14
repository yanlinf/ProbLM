"""
Memory-efficient language modeling via probabilistic counting.

Programmer: fyl
Date: 2018/8/13
"""
import re
import os
import argparse
import pickle
import logging
import frequency_estimation

PUNCS = ',.=[]{}/\\<>!@#$%^&*()-+_|`~"'


def tokenize(line, ngram_size):
    """
    Helper function for tokenizing a single line of text.

    line: str
        a line of text to be tokenized
    ngram_size: int
        (ngram_size - 1) <BOS> tokens will be added at the begining of the sentence

    Returns: list[str]
        a list of tokens
    """
    # lowercase and remove punctuations
    line = line.lower()
    for ch in PUNCS:
        line = line.replace(ch, ' ')
    line = line.strip()

    # replace digits with a <NUM> token
    line = re.sub(r'\b\d+\b', '<NUM>', line)

    tokens = line.split()
    tokens = ['<BOS>'] * (ngram_size - 1) + tokens + ['<EOS>']
    return tokens


class CorpusReader(object):
    """
    Helper class for scanning over the entire corpus and generating ngrams on the fly. We
    assume that a single line of the corpus can still be loaded into the memory even if the
    entire corpus is large.

    Parameters
    ----------
    corpus_path: str
        use the specified corpus to train the model
    ngram_size: int
        ngrams of size ngram_size and (ngram_size - 1) will be generated
    encoding: str, optional (default: utf-8)
        the encoding method of the corpus file
    """

    def __init__(self, corpus_path, ngram_size, encoding='utf-8'):
        self.corpus_path = corpus_path
        self.ngram_size = ngram_size
        self.encoding = encoding
        self.vocabulary = set()
        self.vocab_size = len(self.vocabulary)

    def __iter__(self):
        """
        Returns: generator
            a python generator that yields ngram tuples on the fly
        """
        return self._read_corpus()

    def _read_corpus(self):
        """
        Scanning over the entire corpus and generate ngrams on the fly.

        Returns: generator
            a python generator that yields ngram tuples on the fly
        """
        ngram_size = self.ngram_size
        with open(self.corpus_path, 'r', encoding=self.encoding) as fin:
            for line in fin:
                # tokenize
                words = tokenize(line, ngram_size)

                # update vocabulary
                self.vocabulary |= set(words)

                # skip lines that are too short to produce proper ngrams
                if len(words) < ngram_size - 1:
                    continue

                # yield ngrams of size ngram_size and (ngram_size - 1)
                for offset in range(0, len(words) - ngram_size + 1):
                    yield tuple(words[offset:offset + ngram_size])
                    yield tuple(words[offset:offset + ngram_size - 1])

        self.vocab_size = len(self.vocabulary)


def save_model(model, model_type, vocab_size, filepath):
    """
    Helper function for saving a trained language model to a given location.

    model: frequency_estimation.Sketch object
        the model to be saved
    model_type: str
        Simple / CountSketch / CountMinSketch
    vocab_size: int
        the vocabulary size
    filepath: str
        the location to save the model

    Returns: None
    """
    with open(filepath, 'wb') as fout:
        pickle.dump({
            'type': model_type,
            'counter': model,
            'vocab_size': vocab_size,
            'hash_size': args.hash_size,
            'hash_num': args.hash_num,
            'ngram_size': args.ngram_size,
        }, fout)


def main():
    # choose the counting method base on args
    if args.accurate:
        counter = frequency_estimation.Simple()
        model_type = 'naive'
    elif args.count_sketch:
        counter = frequency_estimation.CountSketch(
            hash_num=args.hash_num, hash_size=args.hash_size)
        model_type = 'count_sketch'
    else:
        counter = frequency_estimation.CountMinSketch(
            hash_num=args.hash_num, hash_size=args.hash_size)
        model_type = 'count_min_sketch'

    # load the input corpus
    reader = CorpusReader(
        args.infile, ngram_size=args.ngram_size, encoding=args.encoding)

    for i, ngram in enumerate(reader):
        logging.debug('processing %s' % str(ngram))
        counter.process(ngram)
        if (i + 1) % 1000000 == 0:
            logging.info('processed %d ngrams' % (i + 1))
            save_model(counter, model_type, reader.vocab_size, args.output)

    # save the model for future evaluation
    save_model(counter, model_type, reader.vocab_size, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('infile',
                        help='the corpus file used to train the model'
                        )
    parser.add_argument('-o', '--output',
                        type=str,
                        default=os.path.join(
                            '.', 'models', 'count_min_sketch'),
                        help='location to save the trained model (default: ./models/count_min_sketch'
                        )
    parser.add_argument('--encoding',
                        type=str,
                        default='utf-8',
                        help='the encoding of the corpus file (default: utf-8)'
                        )
    parser.add_argument('-hn', '--hash_num',
                        type=int,
                        default=32,
                        help='the number of hash functions to use (default: 32)'
                        )
    parser.add_argument('-hs', '--hash_size',
                        type=int,
                        default=65536,
                        help='the size of the hash values'
                        )
    parser.add_argument('-ns', '--ngram_size',
                        type=int,
                        default=5,
                        help='ngrams of size ngram_size - 1 and ngram_size will be counted (default: 5)'
                        )
    parser.add_argument('-v', '--verbose',
                        help='increase verbosity',
                        action='store_const',
                        dest='loglevel',
                        const=logging.INFO,
                        default=logging.WARNING
                        )
    parser.add_argument('-d', '--debug',
                        help='increase verbosity',
                        action='store_const',
                        dest='loglevel',
                        const=logging.DEBUG,
                        )

    # args to determine which counting method to use (deafult:
    # Count_Min_Sketch)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-a', '--accurate',
                       action='store_true',
                       help='use naive accurate counting  (probabilistic counting is used by default)'
                       )
    group.add_argument('--count_sketch',
                       action='store_true',
                       help='use Count_Sketch (the default counter is Count_Min_Sketch)'
                       )

    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel)
    main()
