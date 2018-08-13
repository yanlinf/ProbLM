"""
Convert wikidump corpus in .xml.bz2 format into plain text.

Programmer: fyl
Date: 2018/8/13
"""
from gensim.corpora import WikiCorpus
import argparse
import logging
import multiprocessing


def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    
    # load wiki corpus from a .xml.bz2 file

    wiki = WikiCorpus(args.infile, lemmatize=False, 
            processes=multiprocessing.cpu_count())

    # parse documents from the corpus and write to the output file
    with open(args.outfile, 'w', encoding='utf-8') as fout:
        for i, text in enumerate(wiki.get_texts()):
            fout.write(' '.join(text) + '\n')
            if (i + 1) % 10000 == 0:
                logger.info('Processed %d documents' % (i + 1))
            cnt = i
    logger.info('Finished processing %d documents' % cnt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile',
                        type=str,
                        help='input file (.xml.bz2 format)')
    parser.add_argument('outfile',
                        type=str,
                        help='output file')
    args = parser.parse_args()
    main()
