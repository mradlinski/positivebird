from twitter import *
from .util import get_config

config = get_config()['twitter']

if config['app_auth']:
    bearer_token = oauth2_dance(config['consumer_key'],
                                config['consumer_secret'])

    t = Twitter(auth=OAuth2(
        bearer_token=bearer_token
    ))
else:
    t = Twitter(auth=OAuth(
        config['access_token'],
        config['access_secret'],
        config['consumer_key'],
        config['consumer_secret']
    ))


def get_user_tweets(username):
    tweets = t.statuses.user_timeline(
        screen_name=username,
        count=200,
        trim_user=True,
        exclude_replies=True,
        contributor_details=False,
        include_rts=False
    )
    results = []

    for i, tw in enumerate(tweets):
        results.append((tw['id_str'], tw['text']))

    return results


def get_tweet(id):
    tweet = t.statuses.show(
        id=id,
        trim_user=True,
        include_my_retweet=False,
        include_entities=False
    )

    return tweet['text']
