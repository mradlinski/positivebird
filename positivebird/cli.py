from sys import argv
import os
from .corpus_preprocessing import \
    process_sentiment140, process_sanders, download_sentiment140
from .training import \
    default_operations, save_processed, save_ngrams, save_classifier, perform_test
from .util import get_config


def main():
    argc = len(argv)

    if argc < 2 or argv[1] == 'server':
        from .web import app
        app.run(host='0.0.0.0')
    elif argv[1] == 'download_sentiment140':
        download_sentiment140()
    elif argv[1] == 'corpus':
        if argc < 3:
            process_sentiment140()
            process_sanders()
        elif argv[2] == 'sentiment140':
            process_sentiment140()
        elif argv[2] == 'sanders':
            process_sanders()
        else:
            raise RuntimeError('Invalid corpus name')
    elif argv[1] == 'clean':
        dir = './' + get_config()['bin_folder']
        for f in os.listdir(dir):
            if f != '.gitkeep':
                os.remove('./' + get_config()['bin_folder'] + f)
    elif argv[1] == 'nlp':
        if argc < 3:
            default_operations(False)
        elif argv[2] == 'with_bigrams':
            default_operations(True)
        elif argc < 4:
            raise RuntimeError('No data type specified')
        else:
            with_bigrams = (argc > 4 and argv[4] == 'with_bigrams')
            if argv[2] == 'process':
                save_processed(argv[3])
            elif argv[2] == 'ngrams':
                save_ngrams(argv[3], with_bigrams=with_bigrams)
            elif argv[2] == 'train':
                save_classifier(argv[3], with_bigrams=with_bigrams)
            elif argv[2] == 'test':
                print(perform_test(argv[3], argv[4]))
            else:
                raise RuntimeError('Invalid training command')