# this script contains the core of data augmentation part
import numpy as np
from numpy.fft import rfft, irfft       # fourier transform and inverse function
import pandas as pd
import pywt

# 1D signal distortion through Fourier Transform + random perturbation on the phases
def tsaug(timeserie, sigma=0.2, method='fourier1'):
    '''
    Parameters
    ----------
    timeserie : 1D-ndarray
    sigma : positive scalar accounting for the scale of the perturbation (0 iff no perturbation)
    method : str, either *fourier1*, *fourier2* or *wavelet* accounting the type of transformation

    Returns
    -------
    timeserie2 : 1D-ndarray of the same size of input
    '''

    # since we apply FFT to real signals, we force the length to be an even number
    dim = len(timeserie)
    l = len(timeserie)
    if dim%2:
        ts_last = timeserie[-1]
        timeserie = timeserie[:-1]
        l = len(timeserie)

    # fourier transform on phase
    if method=='fourier1':
        ft = rfft(timeserie)
        perturbation = np.random.normal(0, sigma, size=l//2+1)
        #ftb = ft + perturbation   #noise applied on fourier transform
        #timeserie2 = irfft(ftb)   #inverse fourier transform
        ft_p = ft.real*np.cos(perturbation) - ft.imag*np.sin(perturbation) + 1j*(ft.imag*np.cos(perturbation) +ft.real*np.sin(perturbation))
        timeserie_p = irfft(ft_p)
    # fourier transform on amplitudes, the scale is divided by 10 with respect to the original one
    elif method=='fourier2':
        ft = rfft(timeserie)
        scale_perturbation = np.append(np.ones(l//2+1 - l//4), \
                                    np.exp(np.random.normal(scale = sigma, size = l//4)))
        ft_p = ft*scale_perturbation
        timeserie_p = np.fft.irfft(ft_p)
    elif method=='wavelet':
        cA, cD = pywt.dwt(timeserie, 'db2')     # decomposition
        perturbed_cA = cA + np.random.normal(0, sigma, size=len(cA)) # quasi-local perturbations
        perturbed_cD = cD + np.random.normal(0, 0, size=len(cD)) # small local perturbations
        timeserie_p = pywt.idwt(perturbed_cA, perturbed_cD, 'db2')

    # restore the original size of the vector
    if dim%2:
        timeserie_p = np.append(timeserie_p, ts_last)

    return timeserie_p


# dataframe augmentation
def dfaug(X, sigma=0.2, method='fourier1'):
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
            X_dist[i:i+t_int, j] = tsaug(X_dist[i:i+t_int, j], sigma=sigma, method=method)

    # back to dataframe
    X_dist = pd.DataFrame(X_dist, columns=col)
    return pd.concat([X, X_dist], axis=0)
