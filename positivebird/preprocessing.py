import re
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def preprocess_tweet_phase1(tweet):
    #treat all user handles the same
    tweet = re.sub(r'@[^\s]+', '@USER', tweet)
    #treat all urls the same
    tweet = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', '@URL', tweet)
    #remove multiple spaces
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = tweet.strip()
    return tweet


def preprocess_tweet_phase2(tweet):
    #remove letters repeating more than 3 times in a row
    tweet = re.sub(r'(.)\1\1{1,}', r'\1\1\1', tweet, flags=re.DOTALL)
    tweet = re.sub(r'([a-z])([A-Z])', r'\g<1> \g<2>', tweet)
    tweet = tweet.lower()
    tweet = re.sub(r'[\.,\-#\\]', r' ', tweet, flags=re.DOTALL)
    tweet = tweet.replace('!', ' ! ')
    tweet = tweet.replace('?', ' ? ')
    return tweet


def preprocess_tweet_phase3(tweet):
    words = [process_word(w) for w in tweet.split()]
    words = [w for w in words if w is not None and w != '']
    return words


def preprocess_tweet(tweet):
    tweet = preprocess_tweet_phase1(tweet)
    tweet = preprocess_tweet_phase2(tweet)
    words = preprocess_tweet_phase3(tweet)
    return words


def process_word(word):
    if word in stop_words or re.search(r'^[0-9]*$', word) is not None:
        return None
    word = stemmer.stem(word)
    return word


def check_if_classifiable(tweet):
    return not len(tweet.items()) == 0


def tweet_to_unigrams(tweet, feature_bank):
    return [(w, True) for w in tweet if w in feature_bank]


def tweet_to_bigrams(tweet, feature_bank):
    results = []
    for i in range(len(tweet) - 1):
        if (tweet[i], tweet[i + 1]) in feature_bank:
            results.append(((tweet[i], tweet[i + 1]), True))

    return results


def tweet_to_features(tweet, unigrams, bigrams=None):
    f = tweet_to_unigrams(tweet, unigrams)
    if bigrams is not None:
        f += tweet_to_bigrams(tweet, bigrams)
    return f




