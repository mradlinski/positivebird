import time
import requests
import zipfile
from io import BytesIO

from .util import *
from .twitter_handler import get_tweet

from .preprocessing import preprocess_tweet_phase1

config = get_config()

_src_path = config['data_folder']
_dest_path = config['bin_folder']

_sentiment140_train = {
    'src': config['data_filenames']['sentiment140_train'],
    'dest_p': 'sentiment140_train_p',
    'dest_n': 'sentiment140_train_n'
}

_sentiment140_test = {
    'src': config['data_filenames']['sentiment140_test'],
    'dest_p': 'sentiment140_test_p',
    'dest_n': 'sentiment140_test_n'
}

_sanders = {
    'src': config['data_filenames']['sanders'],
    'dest_p': 'sanders_p',
    'dest_n': 'sanders_n'
}

#html entities to be removed
_sentiment140_to_remove = [
    '&quot;', '&gt;', '&lt;', '&amp;'
]


def preprocess_tweet(line):
    return preprocess_tweet_phase1(line)


def process_source(data_type, fn):
    res_pos, res_neg = fn(_src_path + data_type['src'])
    save_processed(_dest_path + data_type['dest_p'], res_pos)
    save_processed(_dest_path + data_type['dest_n'], res_neg)


def process_sentiment140():
    fn = transform_file_sentiment140
    print('Processing Sentiment140 training set...')
    process_source(_sentiment140_train, fn)
    print('Sentiment140 train processing finished')
    print('Processing Sentiment140 test set...')
    process_source(_sentiment140_test, fn)
    print('Sentiment140 test processing finished')


def process_sanders():
    fn = download_sanders
    print('Downloading Sanders set...')
    process_source(_sanders, fn)
    print('Sanders corpus processing finished')


def save_processed(file, data):
    with open(file, 'w') as f:
        for row in data:
            f.write(row + '\n')


def load_processed(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    return lines


def get_all_tweets(data_type):
    return get_tweets(data_type, 'pos'), get_tweets(data_type, 'neg')


def get_tweets(data_type, label):
    if label == 'pos':
        path = 'dest_p'
    elif label == 'neg':
        path = 'dest_n'
    else:
        raise RuntimeError('Invalid type label (should be pos/neg)')

    if data_type == 'sentiment140_train':
        return load_processed(_dest_path + _sentiment140_train[path])
    elif data_type == 'sentiment140_test':
        return load_processed(_dest_path + _sentiment140_test[path])
    elif data_type == 'sanders':
        return load_processed(_dest_path + _sanders[path])
    else:
        raise RuntimeError('Invalid data type')


def transform_file_sentiment140(src):
    res_pos = []
    res_neg = []

    def process_row(row):
        #filter spammy tweets
        if 'I highly recommends you join www.m2e.asia You can earn money from free shareholder by dividends. Even you do NOTHING!' in row[5]:
            return
        elif 'holidaycarclub' == row[4]:
            return

        sent = int(row[0])

        tweet = row[5]
        for w in _sentiment140_to_remove:
            tweet = tweet.replace(w, '')
        tweet = preprocess_tweet(tweet)

        if sent == 0:
            res_neg.append(tweet)
        #neutral tweets are not loaded, because there are no neutral tweets in the training set
        elif sent == 4:
            res_pos.append(tweet)

    csv_each(src, process_row, encoding='Windows-1252')

    return res_pos, res_neg


def download_sentiment140():
    url = 'http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip'
    request = requests.get(url)
    zip_file_object = zipfile.ZipFile(BytesIO(request.content))

    for file in zip_file_object.namelist():
        f = zip_file_object.open(file)
        content = f.read()
        path = _src_path
        if 'testdata' in file:
            path += _sentiment140_test['src']
        elif 'training' in file:
            path += _sentiment140_train['src']
        dest_file = open(path, 'wb')
        dest_file.write(content)
        dest_file.close()


def download_sanders(src):
    res_pos = []
    res_neg = []

    def process_row(row):
        sent = row[1]

        if sent != 'positive' and sent != 'negative':
            return

        tweet_id = row[2]

        try:
            tweet = get_tweet(tweet_id)
        except:
            return

        tweet = preprocess_tweet(tweet)

        if sent == 'positive':
            res_pos.append(tweet)
        #neutral/irrelevant tweets are not loaded
        elif sent == 'negative':
            res_neg.append(tweet)

        time.sleep(5.5)

    csv_each(src, process_row)

    return res_pos, res_neg