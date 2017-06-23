import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import pickle
import classifier

np.random.seed(25)


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


# read the data & shuffle
df, labels = classifier.read_data()
indices = np.random.permutation(len(labels))
df = df.iloc[indices].reset_index(drop=True)
labels = labels[indices]

# separate to train-test pairs and extract features using TF-IDF
train_idx, test_idx = split_train_test(len(labels), 0.9)
vectorizer = TfidfVectorizer(input='content', ngram_range=(1,2),
                             tokenizer=classifier.title_pos_pairs)
train_data = vectorizer.fit_transform(df.iloc[train_idx].title.values)
test_data = vectorizer.transform(df.iloc[test_idx].title.values)
# define the model we're using
model = svm.LinearSVC
parameters_grid = {'C': [1, 5, 10, 20]}
# train the model while fitting hyperparameters with grid-search and
# cross-validation. then, evaluate the model on the test set
clf = GridSearchCV(model(), param_grid=parameters_grid)
predictions = clf.fit(train_data, labels[train_idx]).predict(test_data)
print('best params:', clf.best_params_)
print('accuracy: %f%%' % (compute_loss(predictions, labels[test_idx])*100))
# save the trained model & vectorizer
pickle.dump(clf, open('svm_model.pkl','wb'))
pickle.dump(vectorizer, open('tf_idf_vectorizer.pkl','wb'))
