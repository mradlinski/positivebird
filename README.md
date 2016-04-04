# PositiveBird - use NLP to check how positive someone is on Twitter

A python web app that uses sentiment analysis on a Twitter user's tweets.

### Tools: 
- _NLTK_ - for various NLP tasks, including sentiment analysis itself (with the Naive Bayes Classifier)
- _twitter_ - Python package used to download tweets
- _flask_ - to expose the NLP functionality via an HTTP API
- _PostgreSQL + psycopg2 + SQLAlchemy_ - to track what users are searching for and hopefully, in the future, allow them to disagree with the analysis and give their own opinion

### Data:
- [_Sentiment140_](http://help.sentiment140.com/for-students) - has a training and a test set, both of which are used. It's pretty gigantic, but the training set is automatically, not hand-annotated. The training set is used for training, the test set is part of the test set.
- [_Sanders Corpus_](http://www.sananalytics.com/lab/twitter-sentiment/) - a smaller, but hand-annotated corpus. Used as the other, bigger part of the test set.
 

Also, the very barebones front-end of the application can be found on the _gh-pages_ branch of this repository. 

### Algorithm:
The sentiment analysis is pretty basic, and heavy inspired by sources such as [this](http://streamhacker.com/2010/05/10/text-classification-sentiment-analysis-naive-bayes-classifier/) and [this](http://ravikiranj.net/posts/2012/code/how-build-twitter-sentiment-analyzer/). Tweets in the corpus are divided into two classes, positive and negative. A Naive Bayes Classifier from NLTK is trained on the Sentiment140 training corpus, and analysis is available through a web API. The API takes a Twitter username, downloads the last 200 tweets from that user and classifies them. The result of that classification is the probability that a tweet is positive. That "positivity" is averaged over all the tweets and returned to the front-end. Then a text result is derived from that probability - either positive, neutral or negative. Neutral results occur when the probability of being positive is close to 50%.

### TODO:
- Allow users to disagree with the classification, save their opinion and possibly use it to further train the classifier

