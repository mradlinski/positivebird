from flask.ext.sqlalchemy import SQLAlchemy

from .util import get_config

db = SQLAlchemy()


def init_db(app):
    app.config.update(get_config()['db'])
    db.init_app(app)
    return db


class DataLabel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tweet_id = db.Column(db.Text)
    ip = db.Column(db.Text)
    session_id = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    label = db.Column(db.Text)

    def __init__(self, tweet_id, label, session_id, ip):
        self.tweet_id = tweet_id
        self.label = label
        self.session_id = session_id
        self.ip = ip


class Visit(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ip = db.Column(db.Text)
    session_id = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())

    def __init__(self, session_id, ip):
        self.session_id = session_id
        self.ip = ip


class UserLookup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ip = db.Column(db.Text)
    session_id = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    user_name = db.Column(db.Text)
    sentiment_result = db.column(db.Float)
    error = db.Column(db.Text)

    def __init__(self, user_name, sentiment_result,
                 session_id, ip, error=None):
        self.user_name = user_name
        self.sentiment_result = sentiment_result
        self.session_id = session_id
        self.ip = ip
        self.error = error
