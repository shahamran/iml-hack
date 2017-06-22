import preprocessing as pre
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

np.random.seed(42)

def split_train_test(num_samples, alpha):
    indices = np.arange(num_samples)
    train_threshold = int(num_samples * alpha)
    train_indices = indices[:train_threshold]
    test_indices = indices[train_threshold:]
    return train_indices, test_indices

def compute_loss(predictions, true_labels):
    m = len(true_labels)
    hits = sum(predictions == true_labels)
    return hits / m

# read the data
df, labels = pre.read_data()

# shuffle it
indices = np.random.permutation(len(labels))
df = df.iloc[indices].reset_index(drop=True)
labels = labels[indices]

# clean and extract features (TF-IDF)
train_idx, test_idx = split_train_test(len(labels), 0.9)
vectorizer = TfidfVectorizer(input='content', strip_accents='unicode',
                             ngram_range=(1,2))
train_data = vectorizer.fit_transform(df.iloc[train_idx].title.values)
test_data = vectorizer.transform(df.iloc[test_idx].title.values)

clf = svm.LinearSVC()
predictions = clf.fit(train_data, labels[train_idx]).predict(test_data)
print('accuracy:', compute_loss(predictions, labels[test_idx]))

# df = pre.clean_data(df)
# clean, word_to_index = pre.extract_features(df)
#
# total_length = len(clean)
# alpha = 0.8
# train_thresh = int(total_length * alpha)
# train_data, train_labels = clean.iloc[:train_thresh], labels[:train_thresh]
# test_data, test_labels = clean.iloc[train_thresh:], labels[train_thresh:]
#
# clf = svm.LinearSVC()
# clf.fit(train_data.values, train_labels)
# predictions = clf.predict(test_data.values)
#
# print('accuracy:', sum(test_labels == predictions) / len(predictions))