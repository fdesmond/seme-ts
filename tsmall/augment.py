# this script contains the core of data augmentation part
import numpy as np
from numpy.fft import rfft, irfft       # fourier transform and inverse function
import pandas as pd
import pywt

# 1D signal distortion through Fourier or Wavelet Transform
def signal_distortion(signal, sigma=0.2, method='fourier2'):
    '''
    Parameters
    ----------
    signal : 1D-ndarray
    sigma : positive scalar accounting for the scale of the perturbation (0 iff no perturbation)
    method : str, either *fourier1*, *fourier2* or *wavelet* accounting the type of transformation

    Returns
    -------
    signal_out : 1D-ndarray of the same size of input

    DESCRIPTION TO ADD
    '''

    # since we apply FFT to real signals, we force the length to be an even number
    dim = len(signal)
    l = len(signal)
    if dim%2:
        signal_last = signal[-1]
        signal = signal[:-1]
        l = len(signal)

    # fourier transform on phase
    if method=='fourier1':
        ft = rfft(signal)
        perturbation = np.random.normal(0, sigma, size=l//2+1)
        #ftb = ft + perturbation   #noise applied on fourier transform
        #signal2 = irfft(ftb)   #inverse fourier transform
        ft_p = ft.real*np.cos(perturbation) - ft.imag*np.sin(perturbation) + 1j*(ft.imag*np.cos(perturbation) +ft.real*np.sin(perturbation))
        signal_out = irfft(ft_p)
    # fourier transform on amplitudes, the scale is divided by 10 with respect to the original one
    elif method=='fourier2':
        ft = rfft(signal)
        scale_perturbation = np.append(np.ones(l//2+1 - l//4), \
                                    np.exp(np.random.normal(scale = sigma, size = l//4)))
        ft_p = ft*scale_perturbation
        signal_out = np.fft.irfft(ft_p)
    elif method=='wavelet':
        cA, cD = pywt.dwt(signal, 'db2')     # decomposition
        perturbed_cA = cA + np.random.normal(0, sigma, size=len(cA)) # quasi-local perturbations
        perturbed_cD = cD + np.random.normal(0, 0, size=len(cD)) # small local perturbations
        signal_out = pywt.idwt(perturbed_cA, perturbed_cD, 'db2')

    # restore the original size of the vector
    if dim%2:
        signal_out = np.append(signal_out, signal_last)

    return signal_out


# dataframe augmentation
def dfaug(X, sigma=0.2, frac_features=0.5, method='fourier2', y_dist=False):
    '''
    Parameters
    ----------
    X : pandas df of size (n, d), the first d-1 columns represent the 1d-signal to be disturbed (see y_dist for last column)
    sigma : positive float64, it tunes the distortion (0 iff no distortion)
    frac_features : float64 in [0,1], fraction of features to perturb for each time window
    method : str, *fourier1* for phase, *fourier2* for amplitude, *wavelet* for wavelet
    y_dist : boolean value, True for changing the last column in X

    Returns
    -------
    X_dist pandas df consisting of a distorted version of X

    DESCRIPTION TO ADD
    # pd.concat([X, X_dist], axis=0)
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
        for j in np.random.choice(range(d-1), int(d*frac_features), replace=False):     # select sqrt(d) different features at random
            X_dist[i:i+t_int, j] = signal_distortion(X_dist[i:i+t_int, j], sigma=sigma, method=method)
            if y_dist: X_dist[i:i+t_int, -1] = signal_distortion(X_dist[i:i+t_int, -1], sigma=sigma, method=method)

    # back to dataframe
    X_dist = pd.DataFrame(X_dist, columns=col)
    return X_dist
