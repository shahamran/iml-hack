import pandas as pd
import numpy as np
from os import path
from load_headlines import load_dataset

# constants
DATA_DIR = '../data'
DATA_FILES = [path.join(DATA_DIR, 'haaretz.csv'),
              path.join(DATA_DIR, 'israelhayom.csv')]

def read_data(filenames=DATA_FILES):
    df, labels = load_dataset(filenames=filenames)
    df = pd.DataFrame(df).rename(columns={0: 'title'})
    return df, np.array(labels)
