"""
Memory-efficient language modeling via probabilistic counting.

Programmer: fyl
Date: 2018/8/13
"""
import argparse
import re
import os
import frequency_estimation
import pickle
from pprint import pprint

PUNCS = ',.=[]{}/\\<>!@#$%^&*()-+_|`~"'

def tokenize(line):
    """
    Helper function for tokenizing a single line of text.

    line: str
        a line of text to be tokenized

    Returns: list[str]
        a list of tokens
    """
    # lowercase and remove punctuations
    line = line.lower()
    for ch in PUNCS:
        line = line.replace(ch, ' ')

    # replace digits with a <NUM> token
    line = re.sub(r'\b\d+\b', '<NUM>', line)

    tokens = line.split()
    tokens.append('<EOS>')
    return tokens



def corpus_reader(corpus_path, ngram_size, encoding='utf-8'):
    """
    Helper function for scanning over the entire corpus and generating ngrams on the fly. We
    assume that a single line of the corpus can still be loaded into the memory even if the
    entire corpus is large.

    corpus_path: str
        use the specified corpus to train the model
    ngram_size: int
        ngrams of size ngram_size and (ngram_size - 1) will be generated
    encoding: str, optional (default: utf-8)
        the encoding method of the corpus file

    Returns: generator
        a python generator that yields ngram tuples on the fly
    """
    with open(corpus_path, 'r', encoding=encoding) as fin:
        for line in fin:
            # tokenize
            words = tokenize(line)

            # skip lines that are too short to produce proper ngrams
            if len(words) < ngram_size - 1:
                continue

            # yield ngrams of size ngram_size and (ngram_size - 1)
            for offset in range(0, len(words) - ngram_size + 1):
                yield tuple(words[offset:offset + ngram_size])
                yield tuple(words[offset:offset + ngram_size - 1])


def save_model(model, model_type, filepath):
    """
    Helper function for saving a trained language model to a given location.

    model: frequency_estimation.Sketch object
        the model to be saved
    model_type: str
        Simple / CountSketch / CountMinSketch
    filepath: str
        the location to save the model

    Returns: None
    """
    with open(filepath, 'wb') as fout:
        pickle.dump({
            'type': model_type,
            'counter': model,
            'hash_size': args.hash_size,
            'hash_num': args.hash_num,
            'ngram_size': args.ngram_size,
        }, fout)


def main():
    # load the input corpus
    reader = corpus_reader(
        args.infile, ngram_size=args.ngram_size, encoding=args.encoding)

    # choose the counting method base on args
    if args.accurate:
        counter = frequency_estimation.Simple()
    elif args.count_sketch:
        counter = frequency_estimation.CountSketch(
            hash_num=args.hash_num, hash_size=args.hash_size)
    else:
        counter = frequency_estimation.CountMinSketch(
            hash_num=args.hash_num, hash_size=args.hash_size)

    for ngram in reader:
        counter.process(ngram)

    # save the model for future evaluation
    save_model(counter, type(counter), args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('infile',
                        help='the corpus file used to train the model')
    parser.add_argument('-o', '--output',
                        type=str,
                        default=os.path.join('.', 'models', 'model'),
                        help='location to save the trained model (default: ./models/count_min_sketch')
    parser.add_argument('--encoding',
                        type=str,
                        default='utf-8',
                        help='the encoding of the corpus file (default: utf-8)')
    parser.add_argument('-hn', '--hash_num',
                        type=int,
                        default=32,
                        help='the number of hash functions to use (default: 32)')
    parser.add_argument('-hs', '--hash_size',
                        type=int,
                        default=65536,
                        help='the size of the hash values')
    parser.add_argument('-ns', '--ngram_size',
                        type=int,
                        default=5,
                        help='ngrams of size ngram_size - 1 and ngram_size will be counted (default: 5)')

    # args to determine which counting method to use (deafult:
    # Count_Min_Sketch)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-a', '--accurate',
                       action='store_true',
                       help='use naive accurate counting  (probabilistic counting is used by default)')
    group.add_argument('--count_sketch',
                       action='store_true',
                       help='use Count_Sketch (the default counter is Count_Min_Sketch)')

    args = parser.parse_args()
    main()
