# sampling and normalizing
import numpy as np

def block_sampling(N, k, l):
    '''select k distinct intervals of size l from 0 to N

    return: a boolean series of length N.
    '''
    seq = list(range(N))
    indices = range(N - (l - 1) * k)
    result = np.array([False] * N)
    offset = 0
    for i in sorted(np.random.choice(indices, k)):
        i += offset
        result[seq[i:i+l]] = 1
        offset += l - 1
    return result


def min_max_normalization(X):
    '''
    input: a pandas series

    return: min max transformation of the time-series
    '''
    return (X - X.min())/(X.max() - X.min())
