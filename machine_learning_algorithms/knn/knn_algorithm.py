from collections import defaultdict
import operator
import numpy as np

from machine_learning_algorithms.knn.knn_distances import *


class KNN:
    def __init__(self, K=3, distance_measure: str = 'euc'):
        self.K = K
        self.measure = self.define_measure(distance_measure)

    def define_measure(self,distance_measure:str):
        if distance_measure == 'euc':
            return euc_dist
        elif distance_measure == 'man':
            return manhattan_dist
        elif distance_measure == 'cheb':
            return chebyshev_dist
        elif distance_measure == 'cosine':
            return cosine_dist
        else:
            return minkowski_dist
        
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        
    def predict(self, X_test):
        predictions = [] 
        for i in range(len(X_test)):
            dist = np.array([self.measure(X_test[i], x_t) for x_t in   
            self.x_train])
            dist_sorted = dist.argsort()[:self.K]
            neigh_count = defaultdict(int)
            for idx in dist_sorted:
                neigh_count[self.y_train[idx]] += 1
            sorted_neigh_count = sorted(neigh_count.items(),    
            key=operator.itemgetter(1), reverse=True)
            predictions.append(sorted_neigh_count[0][0]) 
        return predictions
    