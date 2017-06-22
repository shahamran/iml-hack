import numpy as np
import pandas as pd
from os import path
DATA_DIR = '../../data/2'
DATA_FILE = path.join(DATA_DIR, 'data_50000.pickle')

data = pd.Series(np.load(DATA_FILE))
print(data[:100])