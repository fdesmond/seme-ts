# this script contains the core of data augmentation part
import numpy as np
from numpy.fft import fft, ifft
import pandas as pd

# 1D signal distortion through Fourier Transform + random perturbation on the phases
def tsaug(timeserie, sigma=0.2):
    '''
    Parameters
    ----------
    timeserie : 1D-array
    sigma : scalar

    Returns
    -------
    timeserie2 : 1D-array
    '''
    l = len(timeserie)
    ft = fft(timeserie)     #fourier transform
    perturbation = np.random.normal(0, sigma, size=l)
    #ftb = ft+ perturbation   #noise applied on fourier transform
    #timeserie2 = irfft(ftb)   #inverse fourier transform
    ftb = ft.real*np.cos(perturbation) - ft.imag*np.sin(perturbation) + 1j*(ft.imag*np.cos(perturbation) +ft.real*np.sin(perturbation))
    timeserie2 = ifft(ftb)

    return timeserie2


# dataframe augmentation
def dfaug(X, sigma=0.2):
    '''The input is a pandas dataframe consisting in 1D-signals (the columns). The last column (representing the labels) is not distorted.
    input:
        - X: 2d-ndarray of size (n, d) where the first d-1 columns represent d time series that we want to augment
        - frac: percentage of data with resect to n that we want to generate (frac=1 corresponds to n)

    return: a 2d-ndarray consisting of X concatenated with a distorted version of it
    '''
    n, d = X.shape
    col = X.columns     # save column names for later
    t_int = n // 5      # length of time intervals
    t_points = sorted(np.random.randint(0, n - t_int, 5))       # we select 5 time intervals

    # coping the five blocks into a new 2d-ndarray that will be distorted
    X_dist = np.concatenate((X[t_points[0]:t_points[0]+t_int], X[t_points[1]:t_points[1]+t_int],\
                             X[t_points[2]:t_points[2]+t_int], X[t_points[3]:t_points[3]+t_int],\
                             X[t_points[4]:t_points[4]+t_int]), axis=0)

    # applying distortion to each time-block and for some random features
    for i in [t_int*t for t in range(5)]:
        for j in np.random.choice(range(d-1), int(np.sqrt(d)), replace=False):     # select sqrt(d) different features at random
            X_dist[i:i+t_int, j] = tsaug(X_dist[i:i+t_int, j], sigma=sigma)

    # back to dataframe
    X_dist = pd.DataFrame(X_dist, columns=col)
    return pd.concat([X, X_dist], axis=0)
