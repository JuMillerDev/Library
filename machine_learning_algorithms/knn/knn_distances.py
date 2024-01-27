import numpy as np


def euc_dist(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

def manhattan_dist(x1, x2):
    return np.sum(np.abs(x1 - x2))

def minkowski_dist(x1, x2, p=2):
    return np.power(np.sum(np.power(np.abs(x1 - x2), p)), 1/p)

def chebyshev_dist(x1, x2):
    return np.max(np.abs(x1 - x2))

def cosine_dist(x1, x2):
    dot_product = np.dot(x1, x2)
    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)
    return 1 - (dot_product / (norm_x1 * norm_x2))
