import numpy as np
from collections import Counter

#CART algorithm decision tree
#todo add optimizations
class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None
    
class DecisionTree:

    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        #stopping criteria
        if (depth >= self.max_depth
                or n_labels == 1
                or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        #greedily select the best split according to information gain
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        
        #grow the children that result from the split
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
         # Parent Gini impurity
        parent_gini = self._gini_impurity(y)

        # Calculate Gini impurity for the left and right nodes
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        gini_left = self._gini_impurity(y[left_idxs])
        gini_right = self._gini_impurity(y[right_idxs])

        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)

        # Calculate the weighted sum of Gini impurities for the children
        child_gini = (n_left / n) * gini_left + (n_right / n) * gini_right

        # Calculate the Gini gain (reduction in Gini impurity)
        gini_gain = parent_gini - child_gini
        return gini_gain

    def _gini_impurity(self, y):
        if len(y) == 0:
            return 0.0

        # Calculate the Gini impurity
        p_i = np.array([np.sum(y == c) / len(y) for c in np.unique(y)])
        gini = 1.0 - np.sum(p_i ** 2)
        return gini

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

from keras.datasets import mnist
from sklearn.metrics import accuracy_score

def preprocess_data(x,y,limit):
    x = x.reshape(x.shape[0], -1)
    x = x.astype("float32") / 255
    return x[:limit], y[:limit]

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 10000)
x_test, y_test = preprocess_data(x_test, y_test, 1000)


clf = DecisionTree(max_depth = 10)
clf.fit(x_train, y_train)
y_pred1 = clf.predict(x_test)
acc1 = accuracy_score(y_test, y_pred1)