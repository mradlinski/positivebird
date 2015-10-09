from .training import load_pickled, file_names
from .preprocessing import tweet_to_features, check_if_classifiable, \
    preprocess_tweet_phase1, preprocess_tweet_phase2, preprocess_tweet_phase3


class Classifier():
    def __init__(self, data_type, with_bigrams=False):
        self.data_type = data_type
        self.classifier = load_pickled(file_names[data_type]['classifier'])
        self.unigrams = load_pickled(file_names[data_type]['unigrams'])
        if with_bigrams:
            self.bigrams = load_pickled(file_names[data_type]['bigrams'])
        else:
            self.bigrams = None

    def preprocess(self, tweet):
        return dict(tweet_to_features(
            preprocess_tweet_phase3(
                preprocess_tweet_phase2(
                    preprocess_tweet_phase1(tweet)
                )
            ),
            self.unigrams, self.bigrams
        ))

    def interpret_classification(self, pos_prob):
        if pos_prob > 0.65:
            return 'pos'
        elif pos_prob < 0.35:
            return 'neg'
        else:
            return 'neutral'

    def classify(self, tweet):
        tw = self.preprocess(tweet)
        if not check_if_classifiable(tw):
            return 0.5
        pos_prob = self.classifier.prob_classify(tw).prob('pos')
        return pos_prob

    def classify_many(self, tweets):
        tws = [self.classify(tw) for tw in tweets]
        return tws