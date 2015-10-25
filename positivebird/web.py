import sys
import re
import uuid

import functools
from flask import Flask, jsonify, request
from twitter import TwitterHTTPError

from .twitter_handler import get_user_tweets
from .util import get_config
from .classifier import Classifier
from .db import init_db, DataLabel, Visit, UserLookup

config = get_config()

app = Flask(__name__)
app.debug = config['debug']

db = None
classifier = None
twitter_user_regexp = re.compile(r'(@|)([A-Za-z0-9_]+)')


def get_session_id(generate=False):
    session_id = request.cookies.get('session_id')
    if session_id is None:
        if generate:
            session_id = uuid.uuid4().urn[9:]
        else:
            raise ValueError()
    return session_id


@app.route('/<path:path>', methods=['OPTIONS'])
def options(path):
    return 'OPTIONS'


@app.route('/session', methods=['GET'])
def start_session():
    session_id = get_session_id(generate=True)
    lookup = Visit(session_id, request.remote_addr)
    db.session.add(lookup)
    db.session.commit()

    res = jsonify({
        'session_id': session_id
    })
    res.set_cookie('session_id', session_id)
    return res


@app.route('/label/<tweet_id>/<label>', methods=['POST'])
def label_tweet(tweet_id, label):
    try:
        if re.search(r'^[0-9]+$', tweet_id) is None:
            raise ValueError
        if label not in {'pos', 'neg', 'neutral'}:
            raise ValueError
        session_id = get_session_id()
        labeling = DataLabel(tweet_id, label, session_id, request.remote_addr)
        db.session.add(labeling)
        db.session.commit()
    except Exception as e:
        print(e, file=sys.stderr)
    finally:
        return 'LABEL'


@app.route('/user/<username>', methods=['GET'])
def calc_positivity(username):
    sentiment = 0
    err = None
    response = None
    session_id = None

    try:
        session_id = get_session_id()
        if not twitter_user_regexp.match(username):
            raise ValueError('Invalid twitter username')

        tweets = get_user_tweets(username)

        sentiments = [{
            'sentiment': classifier.classify(tw[1]),
            'id': tw[0]
        } for tw in tweets]

        sent_len = len(sentiments)

        if sent_len == 0:
            raise ValueError('No tweets found')

        sentiment = functools.reduce(
            lambda x, y: x + y['sentiment'],
            sentiments,
            0
        ) / sent_len

        response = {
            'avg_sentiment': sentiment,
            'sentiments': sentiments
        }
    except Exception as e:
        err = e.args[0]
        print(e, file=sys.stderr)
    finally:
        lookup = UserLookup(username, sentiment, session_id,
                            request.remote_addr, err)
        db.session.add(lookup)
        db.session.commit()
        return json_response(response, err)


# @app.route('/text/<text>', methods=['GET'])
# def tweet_pos(text):
#     if len(text) > 200:
#         return json_response(None, 'Text too long')
#     return jsonify({
#         'sentiment': classifier.classify(text)
#     })


@app.after_request
def cors(res):
    res.headers['Access-Control-Allow-Origin'] = 'http://mromnia.github.io'
    res.headers['P3P'] = 'CP="NOI ADM DEV PSAi COM NAV OUR OTRo STP IND DEM"'
    res.headers['Access-Control-Allow-Methods'] = 'GET,POST'
    res.headers['Access-Control-Allow-Credentials'] = 'true'
    res.headers['Access-Control-Allow-Headers'] = 'Content-Type, *'
    return res


def json_response(data, error=None):
    response = jsonify({
        'data': data
    })

    if error is not None:
        response.status_code = 400
    return response


def init():
    global app
    global classifier
    global db
    db = init_db(app)
    classifier = Classifier(config['used_classifier'], config['use_bigrams'])
    return app


def run():
    init()
    app.run(host='0.0.0.0')
