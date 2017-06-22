import pandas as pd
import numpy as np
import sklearn as skl
from os import path
from load_headlines import load_dataset
import re

# constants
DATA_DIR = '../../data/1'
DATA_FILES = [path.join(DATA_DIR, 'haaretz.csv'),
              path.join(DATA_DIR, 'israelhayom.csv')]
CLEANER = re.compile(r'(?:[^a-zA-Z ])+')


def clean_data(df, inplace=False):
    df = df.copy() if inplace is False else df

    def data_cleaner(title):
        title = title.lower()
        title = CLEANER.sub(' ', title)
        return title

    df.loc[:, 'title'] = df.title.apply(data_cleaner)
    return df

def extract_features(df, inplace=False):
    df = df.copy() if inplace is False else df
    words = set()
    for title in df.title.values:
        words = words.union(set(title.split()))
    words_dict = {i: word for i, word in enumerate(words)}
    word_to_index = {word: i for i, word in enumerate(words)}
    df = pd.concat([df, pd.DataFrame(columns=np.arange(len(words)))])

    def phi(row):
        text = row.title
        for word in text.split():
            row[word_to_index[word]] = 1
        return row
    df = df.apply(phi, axis=1).fillna(0).drop('title', 1)
    return df, word_to_index

def read_data(filenames=DATA_FILES):
    df, labels = load_dataset(filenames=filenames)
    df = df.rename(columns={0: 'title'})
    return df, np.array(labels)
