"""
Evaluate the language model by manually typing a sentence and computing its probability. 

Programmer: fyl
Date: 2018/8/13
"""
import argparse
import pickle
import logging
from train import tokenize


def load_model():
    """
    Retore the trained model.

    Returns: (frequency_estimation.Sketch object, ngram_size, vocab_size)
    """
    with open(args.model, 'rb') as fin:
        dic = pickle.load(fin)

    for k, v in dic.items():
        logging.info(k + ' = %r' % v)

    print(list(dic['counter'].counters[0]))
    # print(np.zeros((4, 5)))

    return dic['counter'], dic['ngram_size'], dic['vocab_size']


def main():
    counter, ngram_size, vocab_size = load_model()

    while True:
        line = input('Enter a sentence (EXIT to break):')

        if line == 'EXIT':
            break

        words = tokenize(line, ngram_size)

        probability = 1.
        for offset in range(0, len(words) - ngram_size + 1):
            history = tuple(words[offset:offset + ngram_size - 1])
            joint = tuple(words[offset:offset + ngram_size])

            history_count = counter[history]
            joint_count = counter[joint]

            logging.info(str(history) + '\t count = %d' % history_count)
            logging.info(str(joint) + '\t count = %d' % joint_count)

            # probability with additive smoothing
            probability *= (joint_count + 1) / (history_count + vocab_size)

        print()
        print('------------------------------------------')
        print('Probability: %.40f' % probability)
        print()
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        type=str,
                        help='the trained language model'
                        )
    parser.add_argument('-v', '--verbose',
                        help='increase verbosity',
                        action='store_const',
                        dest='loglevel',
                        const=logging.INFO,
                        default=logging.WARNING
                        )
    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel)
    main()
