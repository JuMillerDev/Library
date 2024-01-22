import numpy as np


def euc_dist(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

def manhattan_dist(x1, x2):
    return np.sum(np.abs(x1 - x2))

def minkowski_dist(x1, x2, p=2):
    return np.power(np.sum(np.power(np.abs(x1 - x2), p)), 1/p)