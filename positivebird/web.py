import re
import functools
from flask import Flask, jsonify
from twitter import TwitterHTTPError

from .twitter_handler import get_user_tweets
from .util import get_config
from .classifier import Classifier


config = get_config()

app = Flask(__name__)
app.debug = config['debug']

classifier = Classifier(config['used_classifier'], config['use_bigrams'])
twitter_user_regexp = re.compile(r'(@|)([A-Za-z0-9_]+)')


@app.route('/user/<username>', methods=['GET'])
def calc_positivity(username):
    if not twitter_user_regexp.match(username):
        return json_response(None, 'Invalid username')

    try:
        tweets = get_user_tweets(username)
    except TwitterHTTPError:
        return json_response(None, 'Twitter error')

    sentiments = [{
        'sentiment': classifier.classify(tw[1]),
        'id': tw[0]
    } for tw in tweets]

    sentiment = functools.reduce(
        lambda x, y: x + y['sentiment'],
        sentiments,
        0
    ) / len(sentiments)

    return json_response({
        'avg_sentiment': sentiment,
        'sentiments': sentiments
    })


@app.route('/text/<text>', methods=['GET'])
def tweet_pos(text):
    if len(text) > 200:
        return json_response(None, 'Text too long')
    return jsonify({
        'sentiment': classifier.classify(text)
    })


@app.after_request
def cors(res):
    res.headers['Access-Control-Allow-Origin'] = '*'
    return res


def json_response(data, error=None):
    response = jsonify({
        'error': error,
        'data': data
    })

    if error is not None:
        response.status_code = 400
    return response