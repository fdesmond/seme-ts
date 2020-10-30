# sampling and normalization
import numpy as np


def block_sampling(N, k, l):
    '''
    Parameters
    ----------
    N : int64, total length of the output vector
    k : int64, number of blocks to select
    l : int64, block length

    Returns
    -------
    Boolean series of length N

    Select k distinct intervals of size l from 0 to N
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
    Parameters
    ----------
    X : 1-dim numpy array

    Returns
    -------
    min-max transformation of the input

    Classical min-max scaling function.
    '''
    return (X - X.min())/(X.max() - X.min())
