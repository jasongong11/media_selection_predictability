import math
import numpy as np

def contains(small, big):
    try:
        big.tobytes().index(small.tobytes())//big.itemsize
        return True
    except ValueError:
        return False

def actual_entropy(l):
    n = len(l)
    sum_gamma = 0

    for i in range(1, n):
        sequence = l[:i]

        for j in range(i+1, n+1):
            s = l[i:j]
            if contains(s, sequence) != True:
                sum_gamma += len(s)
                break

    ae = 1 / (sum_gamma / n) * math.log(n)
    return ae

def unc_entropy(l):
    prob_dist = np.unique(l, return_counts=True)[1]/l.shape[0]
    S_unc = - np.sum(prob_dist * np.log2(prob_dist))
    return S_unc

def rand_entropy(l):
    n = len(l)
    S_rand = np.log2(n)
    return S_rand