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


def clean_data(df):
    df = df.copy()

    def data_cleaner(title):
        title = title.lower()
        title = CLEANER.sub(' ', title)
        return title
    df.loc[:, 'title'] = df.title.apply(data_cleaner)
    words = set()
    for title in df.title.values:
        words = words.union(set(title.split()))
    words_dict = {i: word for i, word in enumerate(words)}
    for i in words_dict:
        df.loc[:, i] = 0

    def phi(row):
        for i in words_dict:
            if words_dict[i] in row.title:
                row[i] = 1

        return row
    df = df.apply(phi, axis=1)
    return df, words_dict

def read_data(filenames=DATA_FILES):
    df, labels = load_dataset(filenames=filenames)
    df = df.rename(columns={0: 'title'})
    return df, labels
