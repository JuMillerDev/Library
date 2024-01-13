from collections import defaultdict
import operator
import numpy as np

from machine_learning_algorithms.knn.knn_distances import euc_dist


class KNN:
    def __init__(self, K=3):
        self.K = K
        
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        
    def predict(self, X_test):
        predictions = [] 
        for i in range(len(X_test)):
            dist = np.array([euc_dist(X_test[i], x_t) for x_t in   
            self.x_train])
            dist_sorted = dist.argsort()[:self.K]
            neigh_count = {}
            for idx in dist_sorted:
                if self.y_train[idx] in neigh_count:
                    neigh_count[self.y_train[idx]] += 1
                else:
                    neigh_count[self.y_train[idx]] = 1
            sorted_neigh_count = sorted(neigh_count.items(),    
            key=operator.itemgetter(1), reverse=True)
            predictions.append(sorted_neigh_count[0][0]) 
        return predictions