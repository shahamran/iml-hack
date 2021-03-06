"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2017

            **  Headline Classifier  **

Author(s):
barak.halle
ransha
===================================================
"""
# imports
import pickle
import pandas as pd
import numpy as np
from load_headlines import load_dataset
import os
# set environment variable for nltk
NLTK_DATA_DIR = os.path.realpath(os.path.join('.', 'nltk'))
os.environ['NLTK_DATA'] = NLTK_DATA_DIR
import nltk

# constants
DATA_DIR = 'data'
DATA_FILES = [os.path.join(DATA_DIR, 'haaretz.csv'),
              os.path.join(DATA_DIR, 'israelhayom.csv')]


def read_data(filenames=DATA_FILES):
    """
    reads the data in the expected format
    :param filenames: the file names of the newspapers
    :return: df (pandas DataFrame), labels (ndarray)
    """
    df, labels = load_dataset(filenames=filenames)
    df = pd.DataFrame(df).rename(columns={0: 'title'})
    return df, np.array(labels)


def title_pos_pairs(title):
    """
    :param title: a headline from a newspaper
    :return: tokenized headline with POS tagging ("word-pos" format)
    """
    tagged_title = nltk.pos_tag(nltk.word_tokenize(title), tagset='universal')
    return ['%s-%s' % tagged_word for tagged_word in tagged_title]


class Classifier(object):

    def __init__(self):
        """
        reads the pickled models
        """
        self.clf = pickle.load(open('svm_model.pkl', 'rb'))
        self.vectorizer = pickle.load(open('tf_idf_vectorizer.pkl', 'rb'))

    def classify(self,X):
        """
        Recieves a list of m unclassified headlines, and predicts for each one
        which newspaper published it.
        :param X: A list of length m containing the headlines' texts (strings)
        :return: y_hat - a binary vector of length m
        """
        transformed_data = self.vectorizer.transform(np.array(X))
        return self.clf.predict(transformed_data)

