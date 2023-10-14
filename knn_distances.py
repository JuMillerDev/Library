import numpy as np


def euc_dist(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))