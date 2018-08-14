"""
Multi-process version of train.py

Programmer: fyl
Date: 2018/8/13
"""
import re
import os
import argparse
import pickle
import itertools
import logging
from multiprocessing import Process, Manager, cpu_count
from train import tokenize, save_model
import frequency_estimation


def line_reader(line, ngram_size, vocabulary):
    # tokenize
    words = tokenize(line, ngram_size)

    # update vocabulary
    vocabulary |= set(words)

    # skip lines that are too short to produce proper ngrams
    if len(words) < ngram_size - 1:
        return

    # yield ngrams of size ngram_size and (ngram_size - 1)
    for offset in range(0, len(words) - ngram_size + 1):
        yield tuple(words[offset:offset + ngram_size])
        yield tuple(words[offset:offset + ngram_size - 1])


def worker(pid, in_queue, out_list, args):
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

    vocabulary = set()

    while True:
        line = in_queue.get(block=True)

        if line is None:
            out_list.append((counter, vocabulary))
            return

        for ngram in line_reader(line, args.ngram_size, vocabulary):
            counter.process(ngram)


def main():
    mangaer = Manager()
    results = mangaer.list()
    work = mangaer.Queue(args.process)

    pool = []
    for i in range(args.process):
        p = Process(target=worker, args=(i, work, results, args))
        p.start()
        pool.append(p)

    with open(args.infile, 'r', encoding=args.encoding) as fin:
        lines = itertools.chain(fin, (None,) * args.process)
        for i, line in enumerate(lines):
            work.put(line)
            if (i + 1) % 100 == 0:
                logging.info('processed %d lines' % (i + 1))

    for p in pool:
        p.join()

    # save the model for future evaluation
    merged_counter, merged_vocab = results[0]
    for counter, vocab in results[1:]:
        merged_counter += counter
        merged_vocab |= vocab
    save_model(merged_counter, '<type?>', len(merged_vocab), args.output, args)
    logging.info('model saved to %s' % args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('infile',
                        help='the corpus file used to train the model'
                        )
    parser.add_argument('-p', '--process',
                        type=int,
                        default=cpu_count(),
                        help='number of processes to use (default: %d)' % cpu_count())
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
                        default=3,
                        help='ngrams of size ngram_size - 1 and ngram_size will be counted (default: 3)'
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

    logging.basicConfig(level=args.loglevel,
                        format='%(asctime)s: %(levelname)s: %(message)s')
    main()
