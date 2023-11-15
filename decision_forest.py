import numpy as np 
from collections import Counter

from decision_tree import DecisionTree

def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    return X[idxs], y[idxs]
def most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common
class RandomForest:
    
    def __init__(self, n_trees=10, min_samples_split=2,
                 max_depth=100, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            print("tree: ", len(self.trees)+1)
            tree = DecisionTree(min_samples_split=self.min_samples_split,
                max_depth=self.max_depth, n_feats=self.n_feats)
            X_samp, y_samp = bootstrap_sample(X, y)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)

from keras.datasets import mnist
from sklearn.metrics import accuracy_score
    
def preprocess_data(x,y,limit):
    x = x.reshape(x.shape[0], -1)
    x = x.astype("float32") / 255
    return x[:limit], y[:limit]

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 10000)
x_test, y_test = preprocess_data(x_test, y_test, 1000)

print("starting")
clf = RandomForest(n_trees = 3)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_train) 
acc1 = accuracy_score(y_train, y_pred)
print("Training Accuracy: ", acc1)

y_pred = clf.predict(x_test)
acc2 = accuracy_score(y_test, y_pred)
print("Testing Acuracy: ", acc2)  