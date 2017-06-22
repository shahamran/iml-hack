import preprocessing as pre
import numpy as np
import sklearn as skl
from sklearn import svm

# read the data
df, labels = pre.read_data()
# shuffle it
indices = np.random.permutation(len(labels))
df = df.iloc[indices].reset_index(drop=True)
labels = labels[indices]

# clean and extract features (word2vec)
df = pre.clean_data(df.iloc[:1000])
labels = labels[:1000]
clean, word_to_index = pre.extract_features(df)

total_length = len(clean)
alpha = 0.8
train_thresh = int(total_length * alpha)
train_data, train_labels = clean.iloc[:train_thresh], labels[:train_thresh]
test_data, test_labels = clean.iloc[train_thresh:], labels[train_thresh:]

clf = svm.SVC()
clf.fit(train_data.values, train_labels)
predictions = clf.predict(test_data.values)

print('accuracy:', sum(test_labels == predictions) / len(predictions))