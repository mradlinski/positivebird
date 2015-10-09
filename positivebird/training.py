import pickle
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
import nltk.classify.util
from nltk import NaiveBayesClassifier
from nltk import MaxentClassifier

from .preprocessing import \
    preprocess_tweet_phase1, preprocess_tweet_phase2, preprocess_tweet_phase3, \
    tweet_to_features
from .corpus_preprocessing import get_all_tweets
from .util import get_config


_dest_path = get_config()['bin_folder']

file_names = {
    'sentiment140_train': {
        'unigrams': 'sentiment140_train_unigrams',
        'bigrams': 'sentiment140_train_bigrams',
        'processed': 'sentiment140_train_processed',
        'classifier': 'sentiment140_train_classifier'
    },
    'sentiment140_test': {
        'unigrams': 'sentiment140_test_unigrams',
        'bigrams': 'sentiment140_test_bigrams',
        'processed': 'sentiment140_test_processed',
        'classifier': 'sentiment140_test_classifier'
    },
    'sanders': {
        'unigrams': 'sanders_unigrams',
        'bigrams': 'sanders_bigrams',
        'processed': 'sanders_processed',
        'classifier': 'sanders_classifier'
    }
}

_best_unigram_cutoff = 5000
_best_bigram_cutoff = 200

_to_remove_sentiment140 = [  # removing some quirks of the sentiment140 data
    '447', 'airfrance',  # mentions of flight 447
    'mahon', 'mcmahon', 'ed mcmahon',  # mentions of Ed McMahon dying
    'farrah', 'farrah fawcett',  # mentions of Farrah Fawcett dying
    'eddings', 'david eddings',  # mentions of David Eddings dying
    'ceci', 'bro ceci',  # mentions of Bro Ceci dying
    'os3',  # mentions of iPhone OS3 not working..? I think
    'fuzzball'  # some random fuzzball spam
]


def default_operations(with_bigrams=False):
    print('Performing full training...')
    save_processed('sentiment140_train')
    print('sentiment140 train processed')
    save_processed('sentiment140_test')
    print('sentiment140 test processed')
    save_processed('sanders')
    print('Sanders processed')
    save_ngrams('sentiment140_train', with_bigrams=with_bigrams)
    print('ngrams found')
    save_classifier('sentiment140_train')
    print('Classifier trained')
    acc1 = perform_test('sentiment140_train', 'sentiment140_test')
    acc2 = perform_test('sentiment140_train', 'sanders')
    print('Accuracy of sentiment140 train set:')
    print('on sentiment140 test = {}'.format(acc1))
    print('on Sanders = {}', acc2)
    print('average = {}', (acc1 + acc2) / 2)


def save_processed(data_type):
    pos_tweets, neg_tweets = get_processed_tweets(data_type)
    save_pickled((pos_tweets, neg_tweets), file_names[data_type]['processed'])


def save_ngrams(data_type, with_unigrams=True, with_bigrams=False):
    pos_tweets, neg_tweets = load_pickled(file_names[data_type]['processed'])

    unigrams, bigrams = get_best_ngrams(pos_tweets, neg_tweets, with_unigrams, with_bigrams)
    save_pickled(unigrams, file_names[data_type]['unigrams'])
    save_pickled(bigrams, file_names[data_type]['bigrams'])


def save_classifier(data_type, with_bigrams=False):
    pos_tweets, neg_tweets = load_pickled(file_names[data_type]['processed'])

    unigrams = load_pickled(file_names[data_type]['unigrams'])

    if with_bigrams:
        bigrams = load_pickled(file_names[data_type]['bigrams'])
    else:
        bigrams = set()

    classifier = train_classifier(pos_tweets, neg_tweets, unigrams, bigrams)
    save_pickled(classifier, file_names[data_type]['classifier'])


def perform_test(train_data_type, test_data_type):
    pos_tweets, neg_tweets = load_pickled(file_names[test_data_type]['processed'])
    unigrams = load_pickled(file_names[train_data_type]['unigrams'])
    bigrams = load_pickled(file_names[train_data_type]['bigrams'])
    classifier = load_pickled(file_names[train_data_type]['classifier'])

    accuracy = test_classifier(classifier, pos_tweets, neg_tweets, unigrams, bigrams)
    return accuracy


def train_classifier(pos_tweets, neg_tweets, unigrams, bigrams):
    train_feats = tweets_to_features(pos_tweets, neg_tweets, unigrams, bigrams)
    classifier = NaiveBayesClassifier.train(train_feats)
    return classifier


def test_classifier(classifier, pos_tweets, neg_tweets, unigrams, bigrams):
    test_feats = tweets_to_features(pos_tweets, neg_tweets, unigrams, bigrams)
    return nltk.classify.util.accuracy(classifier, test_feats)


def tweets_to_features(pos_tweets, neg_tweets, unigrams, bigrams):
    return [
        (dict(tweet_to_features(f, unigrams, bigrams)), 'neg')
        for f in neg_tweets
    ] + [
        (dict(tweet_to_features(f, unigrams, bigrams)), 'pos')
        for f in pos_tweets
    ]


def get_best_ngrams(pos_tweets, neg_tweets, with_unigrams=True, with_bigrams=False):
    words_pos = get_all_unigrams(pos_tweets)
    words_neg = get_all_unigrams(neg_tweets)

    if with_unigrams:
        best_unigrams = get_best_unigrams(
            words_pos,
            words_neg
        )
    else:
        best_unigrams = set()

    if with_bigrams:
        best_bigrams = get_best_bigrams(words_pos + words_neg)
    else:
        best_bigrams = set()

    return best_unigrams, best_bigrams


def get_processed_tweets(data_type):
    pos_tweets, neg_tweets = get_all_tweets(data_type)
    if data_type == 'sentiment140_train':
        pos_tweets = process_tweets_sentiment140_train(pos_tweets)
        neg_tweets = process_tweets_sentiment140_train(neg_tweets)
    else:
        pos_tweets = process_tweets(pos_tweets)
        neg_tweets = process_tweets(neg_tweets)
    return pos_tweets, neg_tweets


def process_tweets(tweets):
    return [preprocess_tweet_phase3(
        preprocess_tweet_phase2(
            preprocess_tweet_phase1(t)
        )
    ) for t in tweets]


def process_tweets_sentiment140_train(tweets):
    results = []
    for t in tweets:
        t = preprocess_tweet_phase2(preprocess_tweet_phase1(t))
        for w in _to_remove_sentiment140:
            t = t.replace(w, '')
        results.append(preprocess_tweet_phase3(t))

    return results


def get_all_unigrams(tweets):
    return [
        w for tw in tweets
            for w in tw
    ]


def get_best_unigrams(words_pos, words_neg):
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()

    for w in words_pos:
        word_fd[w] += 1
        label_word_fd['pos'][w] += 1

    for w in words_neg:
        word_fd[w] += 1
        label_word_fd['neg'][w] += 1

    pos_count = label_word_fd['pos'].N()
    neg_count = label_word_fd['neg'].N()

    total_count = pos_count + neg_count

    word_scores = {}

    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(
            label_word_fd['pos'][word],
            (freq, pos_count),
            total_count
        )
        neg_score = BigramAssocMeasures.chi_sq(
            label_word_fd['neg'][word],
            (freq, neg_count),
            total_count
        )
        word_scores[word] = pos_score + neg_score

    best = sorted(word_scores.items(), key=lambda t: t[1], reverse=True)[:_best_unigram_cutoff]
    bestwords = set([w for w, s in best])

    return bestwords


def get_best_bigrams(words):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, _best_bigram_cutoff)
    return set(bigrams)


def load_pickled(file):
    f = open(_dest_path + file + '.pickle', 'rb')
    data = pickle.load(f)
    f.close()
    return data


def save_pickled(data, file):
    f = open(_dest_path + file + '.pickle', 'wb')
    pickle.dump(data, f, 2)
    f.close()