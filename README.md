# SentimentAnalysis
Creation of Sentiment Analyzer based on IMDB data

### Python 3 Required
To install necessary modules run `pip install -r requirements.txt`

### Necessary Data
In order to train a RNN model for the Sentiment Analysis download the IMDB 
dataset and the GloVe embeddings. You can get them correspondingly at

1. http://ai.stanford.edu/~amaas/data/sentiment/
2. http://nlp.stanford.edu/data/glove.6B.zip

Run from you terminal the following commands

`mkdir data; cd data`

`wget http://ai.stanford.edu/~amaas/data/sentiment/; unzip aclImdb.zip`

`wget http://nlp.stanford.edu/data/glove.6B.zip; unzip glove.6B.zip`

### Training the model
After installing the module dependencies, run `python train_rnn.py` 
