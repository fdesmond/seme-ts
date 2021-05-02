import numpy as np


def block_sampling(
    N: int,
    number_of_blocks: int,
    block_length: int
) -> np.ndarray:
    '''
    Parameters
    ----------
    N : int64, total length of the output vector
    number_of_blocks (int): number of blocks to select
    block_length : int64, block length

    Returns
    -------
    Boolean series of length N

    Select number_of_blocks distinct intervals of size l from 0 to N
    '''
    seq = list(range(N))
    indices = range(N - (block_length - 1) * number_of_blocks)
    result = np.array([False] * N)
    offset = 0
    for i in sorted(np.random.choice(indices, number_of_blocks)):
        i += offset
        result[seq[i:i + block_length]] = 1
        offset += block_length - 1
    return result


def min_max_normalization(X: np.array) -> np.array:
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
